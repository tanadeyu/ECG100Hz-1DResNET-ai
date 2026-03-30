# ECG100Hz-1DResNET-ai Project

## Communication
- **思考**: 英語
- **応答**: 日本語

## ドキュメント作成ルール
- **すべてのドキュメント（MDファイルなど）は日本語で書くこと**

## Project Overview
PTB-XLデータセットを使用した心電図（ECG）の5クラスマルチラベル分類プロジェクト

- **データセット**: PTB-XL (Physikalisch-Technische Bundesanstalt ECG Dataset)
- **モデル**: 1D-ResNet, 1D-Transformer
- **タスク**: 5クラスマルチラベル分類（NORM, MI, STTC, CD, HYP）
- **ライセンス**: MIT License
- **データセットライセンス**: CC BY 4.0

## Python Environment

| ツール | パス |
|--------|------|
| Python | `.conda/env/python.exe` |
| pip | `.conda/env/Scripts/pip.exe` |
| Jupyter | `.conda/env/Scripts/jupyter.exe` |

### 環境情報
- **Python**: 3.11.15
- **PyTorch**: 2.10.0 (CPU)

## Dataset Config
- **データセット**: PTB-XL
- **サンプリング**: 100Hz
- **データ数**: 21,837件（Train: 17,441、Valid: 2,193、Test: 2,203）
- **導出数**: 12-lead ECG
- **信号長**: 10秒（1000サンプル）
- **ラベル数**: 5クラス（NORM, MI, STTC, CD, HYP）- マルチラベル分類
- **データセットURL**: https://physionet.org/content/ptb-xl/1.0.3/

## Important Notes

### ⚠️ Jupyter起動時の注意
**必ずプロジェクトの環境（`.conda/env/`）を使ってください**

```
❌ 誤：WindowsのPythonで起動
   jupyter notebook

✅ 正：プロジェクトの環境で起動
   .conda/env/python.exe -m notebook
```

### ⚠️ 注意・免責事項
- 本プロジェクトは**教育的・ポートフォリオ目的**です
- 実際の医療用途での使用には、専用のシステムと認証が必要です
