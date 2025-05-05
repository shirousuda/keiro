from flask import Flask, request, jsonify, render_template
from route_calculator import RouteCalculator
import logging

# ロギングの設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
route_calculator = RouteCalculator()

# アプリケーション起動時に道路ネットワークを読み込む
logger.debug("Loading road network...")
if not route_calculator.load_network():
    logger.error("Failed to load road network")
else:
    logger.debug("Road network loaded successfully")
    # # 重みづけされた道路を可視化
    # if not route_calculator.visualize_weighted_roads():
    #     logger.warning("Failed to visualize weighted roads")
    # else:
    #     logger.debug("Successfully visualized weighted roads")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_accident_points')
def get_accident_points():
    try:
        # 事故データを取得
        accident_data = route_calculator.accident_data
        if accident_data is None:
            return jsonify({'error': '事故データの読み込みに失敗しました。'}), 500

        # 座標データをリストに変換
        points = []
        for _, row in accident_data.iterrows():
            points.append({
                'lat': row['latitude'],
                'lng': row['longitude']
            })

        return jsonify({'points': points})
    except Exception as e:
        logger.error(f"Error getting accident points: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/search_route', methods=['POST'])
def search_route():
    try:
        data = request.get_json()
        start_point = data['start_point']
        end_point = data['end_point']
        risk_weight = data.get('alpha', 100)  # デフォルト値は100
        
        # リスクの重みの範囲をチェック（0-500）
        risk_weight = max(0, min(500, float(risk_weight)))
        
        # リスクの重みを設定
        route_calculator.risk_weight = risk_weight
        
        result = route_calculator.calculate_route(start_point, end_point)
        if result is None:
            return jsonify({'error': '経路が見つかりませんでした。'})
        
        # 実際に使用されたノードの座標を取得
        start_node_coords = None
        end_node_coords = None
        if 'start_node' in result and 'end_node' in result:
            start_node = result['start_node']
            end_node = result['end_node']
            start_node_coords = {
                'lat': route_calculator.G.nodes[start_node]['y'],
                'lng': route_calculator.G.nodes[start_node]['x']
            }
            end_node_coords = {
                'lat': route_calculator.G.nodes[end_node]['y'],
                'lng': route_calculator.G.nodes[end_node]['x']
            }
        
        return jsonify({
            'best_route': result['best_route'],
            'start_node': start_node_coords,
            'end_node': end_node_coords
        })
        
    except Exception as e:
        logger.error(f"Failed to calculate route: {str(e)}", exc_info=True)
        return jsonify({'error': '経路の計算中にエラーが発生しました。'})

if __name__ == '__main__':
    app.run(debug=True) 