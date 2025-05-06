# 足立区安全経路検索

足立区の道路ネットワークを使用して、事故リスクを考慮した安全な経路を検索するWebアプリケーションです。

## 機能

- 出発地点と到着地点を地図上で指定
- 事故リスクを考慮した経路検索
- リスクの重みを調整可能
- 事故発生箇所の可視化

## デプロイ方法

### Render.com でのデプロイ

1. [Render.com](https://render.com) にアカウントを作成
2. "New +" ボタンをクリックし、"Web Service" を選択
3. GitHub リポジトリと連携
4. 以下の設定を行う：
   - Name: 任意の名前
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Plan: Free

### ローカルでの実行

1. 必要なパッケージをインストール：
```bash
pip install -r requirements.txt
```

2. アプリケーションを起動：
```bash
python app.py
```

3. ブラウザで http://localhost:5000 にアクセス

## 技術スタック

- バックエンド: Python/Flask
- フロントエンド: HTML/CSS/JavaScript
- 地図: OpenStreetMap/Leaflet
- 経路計算: OSMnx
- データ可視化: Folium

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

また、このプロジェクトは以下のライブラリとデータを使用しています：

- OpenStreetMap (ODbL License)
- Leaflet.js (BSD-2-Clause License)
- Bootstrap (MIT License)
- 警察庁交通事故統計データ

各ライブラリとデータの詳細なライセンス情報は[LICENSE](LICENSE)ファイルを参照してください。 