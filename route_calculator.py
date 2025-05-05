import osmnx as ox
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import numpy as np
from typing import List, Tuple, Dict
import logging
from math import radians, sin, cos, sqrt, atan2
import gzip
import shutil
import os
from shapely.strtree import STRtree

logger = logging.getLogger(__name__)

class RouteCalculator:
    def __init__(self, accident_data_path='adachi_data/adachi_honhyo_2023.csv'):
        """
        経路計算クラスの初期化
        
        Args:
            accident_data_path: 事故データのCSVファイルパス
        """
        logger.debug(f"Initializing RouteCalculator with accident data: {accident_data_path}")
        self.accident_data = self._load_accident_data(accident_data_path)
        self.G = None
        self.risk_weight = 100  # リスクの重み（0-500の範囲）
        
    def _dms_to_decimal(self, degrees, minutes, seconds):
        """度分秒形式の座標を10進数に変換"""
        try:
            return float(degrees) + float(minutes)/60 + float(seconds)/3600
        except (ValueError, TypeError):
            logger.warning(f"Failed to convert coordinates: {degrees}, {minutes}, {seconds}")
            return None

    def _load_accident_data(self, file_path):
        """事故データを読み込み、GeoDataFrameとして返す"""
        try:
            logger.debug(f"Loading accident data from {file_path}")
            # CSVファイルを読み込み
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info(f"読み込んだ事故データの総数: {len(df)}")
            
            # 座標は既に10進数形式で保存されているので、そのまま使用
            df['latitude'] = df['latitude'].astype(float)
            df['longitude'] = df['longitude'].astype(float)
            
            # 無効な座標を持つ行を除外
            df = df.dropna(subset=['latitude', 'longitude'])
            logger.info(f"有効な座標を持つ事故データ数: {len(df)}")
            
            # 座標範囲の確認
            logger.info(f"緯度範囲: {df['latitude'].min():.4f}から{df['latitude'].max():.4f}")
            logger.info(f"経度範囲: {df['longitude'].min():.4f}から{df['longitude'].max():.4f}")
            
            # ジオメトリを作成
            geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
            
            # GeoDataFrameを作成
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            logger.debug("Successfully created GeoDataFrame")
            
            return gdf
            
        except Exception as e:
            logger.error(f"Error loading accident data: {str(e)}", exc_info=True)
            return None
    
    def _calculate_edge_accident_score(self, edge_data, u, v):
        """エッジの事故スコアを計算"""
        try:
            # エッジの中心点を取得
            if 'geometry' in edge_data:
                line = edge_data['geometry']
                center = line.interpolate(0.5, normalized=True)
            else:
                # ジオメトリがない場合は始点と終点の中点を使用
                # ノードの座標を使用
                u_coords = (self.G.nodes[u]['x'], self.G.nodes[u]['y'])
                v_coords = (self.G.nodes[v]['x'], self.G.nodes[v]['y'])
                center = Point(
                    (u_coords[0] + v_coords[0]) / 2,
                    (u_coords[1] + v_coords[1]) / 2
                )
            
            # 中心点を投影座標系に変換
            center_proj = gpd.GeoSeries([center], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
            
            # 事故データを投影座標系に変換
            accident_data_proj = self.accident_data.to_crs("EPSG:3857")
            
            # 中心点から100m以内の事故を検索
            nearby_accidents = accident_data_proj[
                accident_data_proj.geometry.distance(center_proj) < 100
            ]
            
            # 事故数をスコアとして返す
            return len(nearby_accidents)
            
        except Exception as e:
            logger.error(f"Error calculating edge accident score: {str(e)}", exc_info=True)
            return 0

    def _assign_accident_scores(self):
        """事故データをループして、近いエッジに事故スコアを付与（足立区のみ）"""
        try:
            # 事故データを投影座標系に変換
            accident_data_proj = self.accident_data.to_crs("EPSG:3857")
            
            # 足立区の境界を取得
            adachi_boundary = ox.geocode_to_gdf('足立区, 東京都, 日本')
            adachi_boundary_proj = adachi_boundary.to_crs("EPSG:3857")
            
            # 車両通行可能な道路のエッジのみを取得
            road_edges = [(u, v, k, data) for u, v, k, data in self.G.edges(keys=True, data=True)
                         if 'highway' in data and data['highway'] not in ['footway', 'pedestrian', 'path', 'cycleway', 'steps']]
            
            logger.info(f"車両通行可能な道路エッジ数: {len(road_edges)}")
            
            # エッジのジオメトリを抽出
            edge_geometries = []
            edge_data_map = {}  # エッジインデックスとデータの対応を保存
            for i, (u, v, k, data) in enumerate(road_edges):
                if 'geometry' in data:
                    geom = data['geometry']
                else:
                    u_coords = (self.G.nodes[u]['x'], self.G.nodes[u]['y'])
                    v_coords = (self.G.nodes[v]['x'], self.G.nodes[v]['y'])
                    geom = LineString([u_coords, v_coords])
                
                # ジオメトリを投影座標系に変換
                geom_proj = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
                edge_geometries.append(geom_proj)
                edge_data_map[i] = (u, v, k, data)
                
                # 事故スコアを初期化
                data['accident_score'] = 0.0
            
            logger.info(f"エッジジオメトリ数: {len(edge_geometries)}")
            
            # 空間インデックスを作成
            edge_tree = STRtree(edge_geometries)
            
            # 各事故地点について処理
            total_accidents = len(accident_data_proj)
            logger.info(f"処理する事故地点の総数: {total_accidents}")
            
            # 足立区内の事故地点数をカウント
            adachi_accidents = 0
            total_edges_updated = 0
            
            # 人身損傷程度に基づく重み付け
            severity_weights = {
                '1': 5.0,  # 死亡事故
                '2': 2.0,  # 負傷事故
                '4': 0.5,  # 損傷なし
                '0': 0.0   # 対象外当事者
            }
            
            for i, (_, accident) in enumerate(accident_data_proj.iterrows(), 1):
                if i % 10 == 0:
                    logger.info(f"事故地点の処理進捗: {i}/{total_accidents} ({i/total_accidents*100:.1f}%)")
                
                accident_point = accident.geometry
                
                # 事故地点が足立区内かチェック
                if not accident_point.within(adachi_boundary_proj.geometry.iloc[0]):
                    continue
                
                adachi_accidents += 1
                
                # 近いエッジを検索（100mバッファ）
                buffer_dist = 100  # メートル
                nearby_indices = edge_tree.query(accident_point.buffer(buffer_dist))
                
                if len(nearby_indices) > 0:
                    # 最も近いエッジを見つける
                    min_dist = float('inf')
                    nearest_edge_idx = None
                    
                    for edge_idx in nearby_indices:
                        edge_geom = edge_geometries[edge_idx]
                        dist = accident_point.distance(edge_geom)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_edge_idx = edge_idx
                    
                    if nearest_edge_idx is not None:
                        # 最も近いエッジにスコアを加算
                        _, _, _, data = edge_data_map[nearest_edge_idx]
                        
                        # 距離に基づく重み付け（近いほど重みが大きい）
                        distance_weight = 1.0 - (min_dist / buffer_dist)
                        distance_weight = max(0.1, min(1.0, distance_weight))  # 重みを0.1から1.0の範囲に制限
                        
                        # 人身損傷程度に基づく重み付け
                        severity_a = str(accident.get("severity_a", "0"))
                        severity_b = str(accident.get("severity_b", "0"))
                        severity_weight = max(
                            severity_weights.get(severity_a, 0.0),
                            severity_weights.get(severity_b, 0.0)
                        )
                        
                        # 最終的なスコアを計算（距離重み × 損傷程度重み）
                        final_weight = distance_weight * severity_weight
                        data['accident_score'] += final_weight
                        
                        total_edges_updated += 1
                        logger.debug(f"事故地点 {i}: 最近接エッジまでの距離 {min_dist:.1f}m, 距離重み {distance_weight:.2f}, 損傷程度重み {severity_weight:.2f}, 最終重み {final_weight:.2f}")
            
            logger.info(f"足立区内の事故地点数: {adachi_accidents}")
            logger.info(f"更新された総エッジ数: {total_edges_updated}")
            
            # 事故スコアを正規化
            max_accidents = max((data.get('accident_score', 0) for _, _, _, data in road_edges), default=0)
            logger.info(f"最大事故スコア: {max_accidents}")
            
            if max_accidents > 0:
                for _, _, _, data in road_edges:
                    data['normalized_risk'] = float(data['accident_score']) / float(max_accidents)
                
                # スコア分布の確認
                scores = [data['normalized_risk'] for _, _, _, data in road_edges]
                logger.info(f"正規化リスクスコアの分布:")
                logger.info(f"  最小値: {min(scores):.3f}")
                logger.info(f"  最大値: {max(scores):.3f}")
                logger.info(f"  平均値: {sum(scores)/len(scores):.3f}")
                logger.info(f"  非ゼロ要素数: {sum(1 for s in scores if s > 0)}")
            else:
                for _, _, _, data in road_edges:
                    data['normalized_risk'] = 0.0
            
            logger.debug("Successfully assigned and normalized accident scores to road edges in Adachi Ward")
            
        except Exception as e:
            logger.error(f"Error assigning accident scores: {str(e)}", exc_info=True)

    def _convert_edge_attributes_to_float(self, G):
        """エッジの属性を数値型に変換"""
        for u, v, k, data in G.edges(keys=True, data=True):
            # 数値属性を変換
            if 'length' in data:
                data['length'] = float(data['length'])
            if 'accident_score' in data:
                data['accident_score'] = float(data['accident_score'])
            if 'custom_cost' in data:
                data['custom_cost'] = float(data['custom_cost'])

    def load_network(self, place_name='Japan'):
        """道路ネットワークを読み込む"""
        try:
            base_graphml_file = 'adachi_base.graphml'
            weighted_graphml_file = 'adachi_weighted.graphml'
            
            # 重み付け済みのグラフが存在する場合はそれを読み込む
            if os.path.exists(weighted_graphml_file):
                logger.debug(f"Loading weighted network from {weighted_graphml_file}")
                self.G = ox.load_graphml(weighted_graphml_file)
                # 属性を数値型に変換
                self._convert_edge_attributes_to_float(self.G)
                return True
            
            # 重み付け前のグラフが存在する場合はそれを読み込む
            if os.path.exists(base_graphml_file):
                logger.debug(f"Loading base network from {base_graphml_file}")
                self.G = ox.load_graphml(base_graphml_file)
                # 属性を数値型に変換
                self._convert_edge_attributes_to_float(self.G)
            else:
                logger.debug("Loading network for Adachi Ward from OSM")
                # 足立区のデータを直接OpenStreetMapから取得
                self.G = ox.graph_from_place('足立区, 東京都, 日本', network_type='drive')
                logger.debug(f"Network loaded from OSM")
                # 重み付け前のグラフを保存
                ox.save_graphml(self.G, base_graphml_file)
                logger.debug(f"Base network saved to {base_graphml_file}")
            
            # エッジの属性を確認し、必要な属性を追加
            for u, v, k, data in self.G.edges(keys=True, data=True):
                # length属性がない場合は計算
                if 'length' not in data or data['length'] is None:
                    # ノードの座標から距離を計算
                    u_coords = (self.G.nodes[u]['x'], self.G.nodes[u]['y'])
                    v_coords = (self.G.nodes[v]['x'], self.G.nodes[v]['y'])
                    length = self.haversine_distance(
                        u_coords[1], u_coords[0],
                        v_coords[1], v_coords[0]
                    )
                    data['length'] = length
                
                # 事故スコアを初期化
                data['accident_score'] = 0.0
                data['normalized_risk'] = 0.0
            
            # 事故スコアを各エッジに付与
            self._assign_accident_scores()
            
            # 重み付け後のグラフを保存
            ox.save_graphml(self.G, weighted_graphml_file)
            logger.debug(f"Weighted network saved to {weighted_graphml_file}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading network: {str(e)}", exc_info=True)
            return False

    def count_possible_routes(self, start_node, end_node, max_nodes=20):
        """
        2点間の可能な経路の数を計算（重みづけなし）
        
        Args:
            start_node: 開始ノード
            end_node: 終了ノード
            max_nodes: 経路の最大ノード数
        
        Returns:
            int: 可能な経路の数
        """
        try:
            # エッジの属性を確認し、必要な属性を追加
            for u, v, k, data in self.G.edges(keys=True, data=True):
                if 'length' not in data or data['length'] is None:
                    # ノードの座標から距離を計算
                    u_coords = (self.G.nodes[u]['x'], self.G.nodes[u]['y'])
                    v_coords = (self.G.nodes[v]['x'], self.G.nodes[v]['y'])
                    length = self.haversine_distance(
                        u_coords[1], u_coords[0],
                        v_coords[1], v_coords[0]
                    )
                    data['length'] = length
                
                # 事故スコアを初期化
                if 'accident_score' not in data:
                    data['accident_score'] = 0
                if 'normalized_risk' not in data:
                    data['normalized_risk'] = 0.0
            
            # 経路をカウント
            route_count = 0
            visited = set()
            
            def dfs(current_node, node_count):
                nonlocal route_count
                
                if current_node == end_node:
                    route_count += 1
                    return
                
                if node_count >= max_nodes:
                    return
                
                visited.add(current_node)
                
                for neighbor in self.G.neighbors(current_node):
                    if neighbor not in visited:
                        dfs(neighbor, node_count + 1)
                
                visited.remove(current_node)
            
            dfs(start_node, 1)
            return route_count
            
        except Exception as e:
            logger.error(f"Error counting routes: {str(e)}", exc_info=True)
            return 0

    def find_connected_nodes(self, start_node, end_node, max_distance=1000):
        """
        開始ノードと終了ノードの近くで連結しているノードを探す
        
        Args:
            start_node: 開始ノード
            end_node: 終了ノード
            max_distance: 最大許容距離（メートル）
        
        Returns:
            tuple: (新しい開始ノード, 新しい終了ノード)
        """
        import networkx as nx
        
        # 開始ノードと終了ノードの座標を取得
        start_coords = (self.G.nodes[start_node]['x'], self.G.nodes[start_node]['y'])
        end_coords = (self.G.nodes[end_node]['x'], self.G.nodes[end_node]['y'])
        
        # 開始ノードと終了ノードの近くのノードを探す
        start_candidates = []
        end_candidates = []
        
        for node in self.G.nodes():
            node_coords = (self.G.nodes[node]['x'], self.G.nodes[node]['y'])
            start_dist = self.haversine_distance(
                start_coords[1], start_coords[0],
                node_coords[1], node_coords[0]
            )
            end_dist = self.haversine_distance(
                end_coords[1], end_coords[0],
                node_coords[1], node_coords[0]
            )
            
            if start_dist < max_distance:
                start_candidates.append((node, start_dist))
            if end_dist < max_distance:
                end_candidates.append((node, end_dist))
        
        # 距離でソート
        start_candidates.sort(key=lambda x: x[1])
        end_candidates.sort(key=lambda x: x[1])
        
        # 連結しているノードのペアを探す
        for s, _ in start_candidates:
            for e, _ in end_candidates:
                try:
                    if nx.has_path(self.G, s, e):
                        return s, e
                except:
                    continue
        
        return start_node, end_node

    def calculate_route(self, start_point, end_point):
        """2点間の経路を計算（リスクを考慮）"""
        if self.G is None:
            logger.error("Network not loaded")
            return None
            
        try:
            logger.debug(f"Calculating route from {start_point} to {end_point}")
            
            # 道路ノードのセットを作成
            road_nodes = set()
            for u, v, k, data in self.G.edges(keys=True, data=True):
                if 'highway' in data and data['highway'] not in ['footway', 'pedestrian', 'path', 'cycleway', 'steps']:
                    road_nodes.add(u)
                    road_nodes.add(v)
            
            logger.debug(f"Found {len(road_nodes)} road nodes")
            
            # 最も近い道路ノードを見つける
            start_node = None
            end_node = None
            min_start_dist = float('inf')
            min_end_dist = float('inf')
            
            for node in road_nodes:
                node_coords = (self.G.nodes[node]['x'], self.G.nodes[node]['y'])
                start_dist = self.haversine_distance(
                    start_point[0], start_point[1],
                    node_coords[1], node_coords[0]
                )
                end_dist = self.haversine_distance(
                    end_point[0], end_point[1],
                    node_coords[1], node_coords[0]
                )
                
                if start_dist < min_start_dist:
                    min_start_dist = start_dist
                    start_node = node
                if end_dist < min_end_dist:
                    min_end_dist = end_dist
                    end_node = node
            
            if start_node is None or end_node is None:
                logger.error("Could not find valid road nodes for start or end points")
                return None
            
            logger.debug(f"Found nearest road nodes: {start_node} -> {end_node}")
            
            # 経路の存在確認
            try:
                # 重みなしで経路が存在するか確認（networkxのshortest_pathを使用）
                import networkx as nx
                route = nx.shortest_path(self.G, start_node, end_node)
                if route:
                    logger.debug("Found path without weights")
                else:
                    logger.error("No path exists between start and end nodes")
                    # 連結しているノードを探す
                    start_node, end_node = self.find_connected_nodes(start_node, end_node)
                    logger.debug(f"Trying alternative nodes: {start_node} -> {end_node}")
            except Exception as e:
                logger.error(f"Error checking path existence: {str(e)}")
                # 連結しているノードを探す
                start_node, end_node = self.find_connected_nodes(start_node, end_node)
                logger.debug(f"Trying alternative nodes: {start_node} -> {end_node}")
            
            if start_node == end_node:
                logger.warning("Start and end nodes are the same, trying to find alternative nodes")
                start_candidates = []
                end_candidates = []
                
                for node in road_nodes:
                    node_data = self.G.nodes[node]
                    start_dist = self.haversine_distance(
                        start_point[0], start_point[1],
                        node_data['y'], node_data['x']
                    )
                    end_dist = self.haversine_distance(
                        end_point[0], end_point[1],
                        node_data['y'], node_data['x']
                    )
                    start_candidates.append((node, start_dist))
                    end_candidates.append((node, end_dist))
                
                start_candidates.sort(key=lambda x: x[1])
                end_candidates.sort(key=lambda x: x[1])
                
                for s, _ in start_candidates[:5]:
                    for e, _ in end_candidates[:5]:
                        if s != e:
                            start_node = s
                            end_node = e
                            break
                    if start_node != end_node:
                        break
            
            logger.debug(f"Using road nodes: {start_node} -> {end_node}")
            
            # 一時的にカスタムコストを計算
            for u, v, k, data in self.G.edges(keys=True, data=True):
                length = float(data['length'])
                normalized_risk = float(data['normalized_risk'])
                # コストを計算（距離 + リスクの重み付け）
                # リスクコストは距離に依存しない
                data['custom_cost'] = length + self.risk_weight * normalized_risk
            
            # リスクを考慮した最短経路を計算
            route = ox.shortest_path(self.G, start_node, end_node, weight='custom_cost')
            
            if route is None:
                logger.error("No valid route found")
                return None
            
            # 経路の座標を取得
            route_coords = []
            total_distance = 0
            total_risk = 0
            
            for j in range(len(route)-1):
                node1 = self.G.nodes[route[j]]
                node2 = self.G.nodes[route[j+1]]
                route_coords.append([
                    float(node1['y']),
                    float(node1['x'])
                ])
                # エッジのデータを取得
                edge_data = self.G.get_edge_data(route[j], route[j+1])
                if edge_data:
                    total_risk += edge_data[0]['accident_score']
                total_distance += self.haversine_distance(
                    float(node1['y']), float(node1['x']),
                    float(node2['y']), float(node2['x'])
                )
            
            last_node = self.G.nodes[route[-1]]
            route_coords.append([
                float(last_node['y']),
                float(last_node['x'])
            ])
            
            # 距離とリスクの計算
            distance_km = total_distance / 1000
            risk_density = total_risk / distance_km if distance_km > 0 else 0
            
            # リスク効率の計算（リスク密度の逆数）
            # リスク密度が高いほど効率が悪い（値が小さい）ことを示す
            # リスク密度が0の場合は大きな値（1e10）を返す
            risk_efficiency = 1.0 / risk_density if risk_density > 0 else 1e10
            
            route_info = {
                'route': route_coords,
                'distance': distance_km,
                'risk_score': total_risk,
                'risk_density': risk_density,
                'risk_efficiency': risk_efficiency,
                'used_risk_weight': self.risk_weight
            }
            
            logger.debug(f"Route calculated with distance: {distance_km:.2f}km, risk score: {total_risk}")
            return {
                'best_route': route_info,
                'start_node': start_node,
                'end_node': end_node
            }
            
        except Exception as e:
            logger.error(f"Error calculating route: {str(e)}", exc_info=True)
            return None

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """2点間の距離をメートル単位で計算（ハバーサイン公式）"""
        R = 6371000  # 地球の半径（メートル）
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c
        
        return distance

    def visualize_weighted_roads(self, output_file='weighted_roads.html'):
        """
        重みづけされた道路を地図上で可視化
        
        Args:
            output_file: 出力するHTMLファイル名
        """
        try:
            # 既にファイルが存在する場合は処理をスキップ
            if os.path.exists(output_file):
                logger.info(f"{output_file} は既に存在します。スキップします。")
                return True

            import folium
            from folium.plugins import HeatMap
            
            # 重みづけされた道路のエッジをGeoDataFrameに変換
            edges_data = []
            edge_geometries = []  # エッジのジオメトリを保存するリスト
            edge_data_map = {}    # エッジインデックスとデータの対応を保存
            
            for i, (u, v, k, data) in enumerate(self.G.edges(keys=True, data=True)):
                if 'highway' in data and data['highway'] not in ['footway', 'pedestrian', 'path', 'cycleway', 'steps']:
                    if 'geometry' in data:
                        line = data['geometry']
                    else:
                        u_coords = (self.G.nodes[u]['x'], self.G.nodes[u]['y'])
                        v_coords = (self.G.nodes[v]['x'], self.G.nodes[v]['y'])
                        line = LineString([u_coords, v_coords])
                    
                    # 数値を確実にfloat型に変換
                    accident_score = float(data.get('accident_score', 0))
                    normalized_risk = float(data.get('normalized_risk', 0))
                    
                    edges_data.append({
                        'geometry': line,
                        'accident_score': accident_score,
                        'normalized_risk': normalized_risk,
                        'highway': data.get('highway', '')
                    })
                    
                    # ジオメトリを投影座標系に変換して保存
                    geom_proj = gpd.GeoSeries([line], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
                    edge_geometries.append(geom_proj)
                    edge_data_map[i] = (u, v, k, data)
            
            gdf = gpd.GeoDataFrame(edges_data, crs="EPSG:4326")
            
            # 足立区の中心座標を取得
            adachi_center = ox.geocode_to_gdf('足立区, 東京都, 日本').geometry.unary_union.centroid
            center_lat, center_lon = adachi_center.y, adachi_center.x
            
            # 地図を作成
            m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
            
            # 事故スコアに基づいて色を設定
            def get_color(score):
                score = float(score)  # 確実にfloat型に変換
                if score == 0:
                    return '#808080'  # グレー（事故なし）
                elif score < 1:
                    return '#90EE90'  # 薄緑（低リスク）
                elif score < 3:
                    return '#FFD700'  # 黄色（中リスク）
                else:
                    return '#FF4500'  # 赤（高リスク）
            
            # 道路を描画
            for _, row in gdf.iterrows():
                if row['accident_score'] > 0:  # 事故がある道路のみ
                    color = get_color(row['accident_score'])
                    folium.PolyLine(
                        locations=[[coord[1], coord[0]] for coord in row['geometry'].coords],
                        color=color,
                        weight=3,
                        opacity=0.8,
                        popup=f"事故スコア: {row['accident_score']:.2f}<br>正規化リスク: {row['normalized_risk']:.2f}<br>道路種別: {row['highway']}"
                    ).add_to(m)
            
            # 事故地点を描画
            # 空間インデックスを作成（エッジ用）
            edge_tree = STRtree(edge_geometries)
            
            # 事故データを投影座標系に変換
            accident_data_proj = self.accident_data.to_crs("EPSG:3857")
            
            # アニメーション用のJavaScriptを追加
            animation_js = """
            <script>
            function animateMarker(marker) {
                let opacity = 0.7;
                let increasing = false;
                
                function updateOpacity() {
                    if (increasing) {
                        opacity += 0.05;
                        if (opacity >= 0.7) {
                            opacity = 0.7;
                            increasing = false;
                        }
                    } else {
                        opacity -= 0.05;
                        if (opacity <= 0.2) {
                            opacity = 0.2;
                            increasing = true;
                        }
                    }
                    marker.setStyle({fillOpacity: opacity});
                }
                
                setInterval(updateOpacity, 100);
            }
            </script>
            """
            m.get_root().html.add_child(folium.Element(animation_js))
            
            # 各事故地点について処理
            for _, accident in accident_data_proj.iterrows():
                # 近くのエッジを検索（100mバッファ）
                nearby_indices = edge_tree.query(accident.geometry.buffer(100))
                
                if len(nearby_indices) > 0:  # 配列の長さをチェック
                    # 最も高いスコアを取得
                    max_score = max(edge_data_map[idx][3]['accident_score'] for idx in nearby_indices)
                    
                    # スコアに基づいて色を設定
                    if max_score == 0:
                        color = '#808080'  # グレー
                    elif max_score < 1:
                        color = '#90EE90'  # 薄緑
                    elif max_score < 3:
                        color = '#FFD700'  # 黄色
                    else:
                        color = '#FF4500'  # 赤
                    
                    # マーカーを作成
                    marker = folium.CircleMarker(
                        location=[accident.geometry.y, accident.geometry.x],
                        radius=5,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        popup=f'事故地点<br>緯度: {accident.geometry.y:.6f}<br>経度: {accident.geometry.x:.6f}<br>事故スコア: {max_score:.2f}<br>当事者Aの損傷程度: {self.get_severity_text(accident.get("severity_a"))}<br>当事者Bの損傷程度: {self.get_severity_text(accident.get("severity_b"))}'
                    )
                    
                    # マーカーを地図に追加
                    marker.add_to(m)
                    
                    # アニメーション用のJavaScriptを追加
                    marker_js = f"""
                    <script>
                    (function() {{
                        var marker = document.querySelector('path[fill="{color}"]');
                        if (marker) {{
                            animateMarker(marker);
                        }}
                    }})();
                    </script>
                    """
                    m.get_root().html.add_child(folium.Element(marker_js))
            
            # 凡例を追加
            legend_html = '''
            <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px">
            <p><strong>事故スコア</strong></p>
            <p><span style="color: #808080;">■</span> 事故なし</p>
            <p><span style="color: #90EE90;">■</span> 低リスク (0-1)</p>
            <p><span style="color: #FFD700;">■</span> 中リスク (1-3)</p>
            <p><span style="color: #FF4500;">■</span> 高リスク (3以上)</p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # 地図を保存
            m.save(output_file)
            logger.debug(f"Map saved to {output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing weighted roads: {str(e)}", exc_info=True)
            return False
    
    def get_severity_text(self, severity):
        severity_map = {
            '1': '死亡',
            '2': '負傷',
            '4': '損傷なし',
            '0': '対象外当事者'
        }
        return severity_map.get(str(severity), '不明')

if __name__ == '__main__':
    # ロギングの設定
    logging.basicConfig(level=logging.INFO)
    
    # RouteCalculatorのインスタンスを作成
    calculator = RouteCalculator()
    
    # 道路ネットワークを読み込む
    calculator.load_network('足立区, 東京都, 日本')
    
    # 事故スコアを計算
    calculator._assign_accident_scores()
    
    # 地図を生成
    calculator.visualize_weighted_roads() 