import pandas as pd
import os
from collections import Counter

def convert_coordinate(x, is_longitude=False):
    """座標文字列を10進数度に変換
    
    Args:
        x: 座標文字列（例: 431007628 -> 43度10分07.628秒）
        is_longitude: 経度の場合True
    """
    try:
        # 文字列に変換して9桁になるまで左を0で埋める
        x_str = str(int(x)).zfill(9)
        
        # 度、分、秒を抽出
        if is_longitude:
            degrees = int(x_str[0:3])  # 経度は3桁
            minutes = int(x_str[3:5])
            seconds = float(x_str[5:7] + '.' + x_str[7:])
        else:
            degrees = int(x_str[0:2])  # 緯度は2桁
            minutes = int(x_str[2:4])
            seconds = float(x_str[4:6] + '.' + x_str[6:])
        
        # 10進数度に変換
        decimal = degrees + minutes/60 + seconds/3600
        
        return decimal
    except Exception as e:
        print(f"座標変換エラー: {x} -> {str(e)}")
        return None

def extract_adachi_data(input_file, output_file):
    """
    警察庁の交通事故データから足立区のデータを抽出
    
    Args:
        input_file: 入力ファイルパス
        output_file: 出力ファイルパス
    """
    print(f"\n{input_file}を処理中...")
    
    # CSVファイルを読み込み
    df = pd.read_csv(input_file, encoding='shift-jis')
    
    # 基本情報の出力
    print(f"総レコード数: {len(df)}")
    
    # 東京都のデータを抽出
    tokyo_data = df[df['都道府県コード'] == 30]  # 30: 東京都
    print(f"東京都のレコード数: {len(tokyo_data)}")
    
    # 足立区の警察署コード
    adachi_police_stations = [667, 668, 669, 670]  # 千住、西新井、綾瀬、竹の塚
    
    # 足立区のデータを抽出
    adachi_data = df[
        (df['都道府県コード'] == 30) & 
        (df['警察署等コード'].isin(adachi_police_stations))
    ]
    
    if not adachi_data.empty:
        # 必要な列のみを選択
        selected_columns = [
            '発生日時　　年', '発生日時　　月', '発生日時　　日',
            '発生日時　　時', '発生日時　　分',
            '地点　緯度（北緯）', '地点　経度（東経）',
            '事故内容', '死者数', '負傷者数',
            '道路形状', '事故類型', '警察署等コード',
            '人身損傷程度（当事者A）', '人身損傷程度（当事者B）'
        ]
        
        adachi_data = adachi_data[selected_columns]
        
        # 列名を簡潔に
        column_mapping = {
            '発生日時　　年': 'year',
            '発生日時　　月': 'month',
            '発生日時　　日': 'day',
            '発生日時　　時': 'hour',
            '発生日時　　分': 'minute',
            '地点　緯度（北緯）': 'latitude',
            '地点　経度（東経）': 'longitude',
            '事故内容': 'accident_type',
            '死者数': 'fatalities',
            '負傷者数': 'injuries',
            '道路形状': 'road_shape',
            '事故類型': 'accident_category',
            '警察署等コード': 'police_station_code',
            '人身損傷程度（当事者A）': 'severity_a',
            '人身損傷程度（当事者B）': 'severity_b'
        }
        adachi_data = adachi_data.rename(columns=column_mapping)
        
        # 警察署名を追加
        station_names = {
            667: '千住',
            668: '西新井',
            669: '綾瀬',
            670: '竹の塚'
        }
        adachi_data['police_station_name'] = adachi_data['police_station_code'].map(station_names)
        
        # 座標を10進数度に変換
        adachi_data['latitude'] = adachi_data['latitude'].apply(lambda x: convert_coordinate(x, False))
        adachi_data['longitude'] = adachi_data['longitude'].apply(lambda x: convert_coordinate(x, True))
        
        # 無効な座標を持つ行を削除
        adachi_data = adachi_data.dropna(subset=['latitude', 'longitude'])
        
        # 足立区の座標範囲でフィルタリング
        # 足立区の概ねの座標範囲
        adachi_bounds = {
            'lat_min': 35.73,  # 足立区の最南端
            'lat_max': 35.82,  # 足立区の最北端
            'lon_min': 139.73, # 足立区の最西端
            'lon_max': 139.86  # 足立区の最東端
        }
        
        # 座標範囲でフィルタリング
        adachi_data = adachi_data[
            (adachi_data['latitude'] >= adachi_bounds['lat_min']) &
            (adachi_data['latitude'] <= adachi_bounds['lat_max']) &
            (adachi_data['longitude'] >= adachi_bounds['lon_min']) &
            (adachi_data['longitude'] <= adachi_bounds['lon_max'])
        ]
        
        # 結果を保存
        adachi_data.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n足立区のデータ {len(adachi_data)}件を{output_file}に保存しました")
        
        # 警察署別の件数を表示
        print("\n警察署別の事故件数:")
        station_counts = adachi_data['police_station_name'].value_counts()
        for station, count in station_counts.items():
            print(f"{station}: {count}件")
        
        # 座標範囲の確認
        print("\n座標範囲:")
        print(f"緯度: {adachi_data['latitude'].min():.4f}°から{adachi_data['latitude'].max():.4f}°")
        print(f"経度: {adachi_data['longitude'].min():.4f}°から{adachi_data['longitude'].max():.4f}°")
    else:
        print(f"\n{input_file}に足立区のデータが見つかりませんでした")

def main():
    # 入力ファイルのリスト
    input_files = [
        'honhyo_2019.csv',
        'honhyo_2020.csv',
        'honhyo_2021.csv',
        'honhyo_2022.csv',
        'honhyo_2023.csv'
    ]
    
    # 出力ディレクトリの作成
    output_dir = 'adachi_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # 各ファイルを処理
    for input_file in input_files:
        if os.path.exists(input_file):
            output_file = os.path.join(output_dir, f'adachi_{input_file}')
            extract_adachi_data(input_file, output_file)
        else:
            print(f"ファイルが見つかりません: {input_file}")

if __name__ == '__main__':
    main() 