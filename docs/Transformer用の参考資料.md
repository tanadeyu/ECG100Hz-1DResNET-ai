# Transformer用の参考資料

## 要点

- **Transformer**: 自己注意機構を使った深層学習モデル
- **1D信号処理**: 音声、時系列、ECGなどに応用可能
- **セグメント分割**: 信号を分割してトークンのように扱う
- **埋め込み**: 各セグメントを固定次元のベクトルに変換
- **位置エンコーディング**: セグメントの順序情報を追加
- **GPU vs CPU**: GPUは計算、CPUはデータ読み込みで役割分担

---

## Transformerとは

### 定義

**自己注意機構（Self-Attention）を中心とした深層学習アーキテクチャ**

### 特徴

| 項目 | 説明 |
|------|------|
| **自己注意** | すべてのトークン間の関係を計算 |
| **並列処理** | RNNと違って全体を並列計算可能 |
| **長距離依存** | 離れた位置の関係も学習可能 |
| **汎用性** | テキスト、画像、音声、時系列に応用可能 |

### 基本構造

```
入力 → 埋め込み層 → 位置エンコーディング → Transformer Encoder × N → 出力
                     ↑                      ↑
                 特徴抽出              自己注意機構
```

---

## 1D信号処理への応用

### 共通概念

```
BERT:    テキスト   → トークン分割 → 埋め込み(768次元) → Transformer
ViT:     画像      → パッチ分割  → 埋め込み(768次元) → Transformer
1D信号:  音声/ECG  → セグメント分割 → 埋め込み(256次元) → Transformer
```

### ECGへの応用

```
ECG信号 (12チャンネル × 1000サンプル)
    ↓
100サンプルずつセグメント分割（10個）
    ↓
各セグメントを256次元ベクトルに埋め込み（1D CNN）
    ↓
位置エンコーディング追加
    ↓
Transformer Encoder（6層、8ヘッド）
    ↓
分類ヘッド → 5クラス出力
```

---

## ResNet vs Transformer

### 比較

| 項目 | ResNet | Transformer |
|------|--------|-------------|
| **得意なパターン** | 局所的な特徴 | 大域的な依存関係 |
| **計算量** | 少ない | 多い（自己注意） |
| **データ量** | 少量でもOK | 多量でより効果的 |
| **長距離依存** | 苦手 | 得意 |
| **パラメータ数** | 多い（約870万） | 少ない（約200万） |
| **1Epoch時間** | 約7秒 | 約5秒（最適化時） |

### 使い分けの目安

| データ数 | 推奨モデル | 理由 |
|---------|----------|------|
| **1,000〜5,000件** | ResNet | Transformerで過学習のリスク |
| **5,000〜20,000件** | ResNet or Transformer | 両方候補 |
| **20,000件以上** | **Transformer推奨** | データ量十分で効果発揮 |

**PTB-XL（21,837件）はTransformerに適したデータ量です。**

---

## Transformer1Dモデルの実装

### アーキテクチャ

```python
class Transformer1D(nn.Module):
    def __init__(self, in_channels=12, num_classes=5,
                 d_model=256,      # 埋め込み次元
                 nhead=8,          # マルチヘッドアテンションのヘッド数
                 num_layers=6,     # Transformer Encoderの層数
                 dim_feedforward=1024,  # フィードフォワード層の次元
                 segment_len=100,  # セグメント長
                 dropout=0.3):
```

### 各層の役割

| 層 | 役割 | 入力 → 出力 |
|----|------|------------|
| **セグメント埋め込み** | 1D CNNで特徴抽出 | (batch, 12, 1000) → (batch, 256, 1000) |
| **セグメント分割** | 100サンプルずつ平均プーリング | (batch, 256, 1000) → (batch, 256, 10) |
| **位置エンコーディング** | 順序情報を追加 | (batch, 10, 256) → (batch, 10, 256) |
| **Transformer Encoder** | 自己注意で特徴抽出 | (batch, 10, 256) → (batch, 10, 256) |
| **分類ヘッド** | クラス分類 | (batch, 256) → (batch, 5) |

### batch_first=Trueの重要性

```python
# 推奨（警告なし）
encoder_layer = nn.TransformerEncoderLayer(
    d_model=256,
    nhead=8,
    batch_first=True  # (batch, seq, feature)形式
)

# 非推奨（警告あり）
encoder_layer = nn.TransformerEncoderLayer(
    d_model=256,
    nhead=8,
    batch_first=False  # (seq, batch, feature)形式
)
```

---

## ハイパーパラメータ設定

### RTX 3060 (12GB) 推奨設定

| 項目 | 推奨値 | 説明 |
|------|--------|------|
| **BATCH_SIZE** | 128〜256 | 大きいほどGPU利用率向上 |
| **LR** | 5e-4 | ResNetより高め |
| **D_MODEL** | 256 | 768だとメモリ不足 |
| **NHEAD** | 8 | d_modelの約1/32 |
| **NUM_LAYERS** | 6 | BERT-baseは12層 |
| **SEGMENT_LEN** | 100 | 1000/100=10セグメント |
| **DROPOUT** | 0.3 | 過学習対策 |

### 学習率スケジューラ

| スケジューラ | 特徴 |
|-------------|------|
| **OneCycleLR** | 推奨。学習安定、収束早い |
| CosineAnnealingWarmRestarts | 学習率が頻繁に変動 |
| ReduceLROnPlateau | シンプルだが効果は穏やか |

### OneCycleLRの設定

```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR * 10,      # 5e-3
    epochs=EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,       # 前半30%で増加
    anneal_strategy='cos'
)
```

---

## 学習の最適化

### Batch Sizeの影響

| Batch Size | バッチ数/Epoch | GPU利用率 | 1Epoch時間 |
|------------|---------------|----------|-----------|
| 64 | 273 | 60〜70% | 9秒 |
| 128 | 137 | 85〜95% | 5秒 |
| 256 | 69 | 95〜100% | 4秒 |

### num_workersの設定

| CPUコア数 | 推奨num_workers |
|----------|-----------------|
| 2〜4コア | 2 |
| 4〜8コア | 3〜4 |
| 8コア以上 | 4〜6 |

**Windowsでは多すぎると逆効果（multiprocessingのオーバーヘッド）**

### GPU vs CPUの役割分担

```
CPU（num_workers=2）:
- データ読み込み（HDD/SSD → メモリ）
- 前処理（正規化、分割）
- GPUへの転送準備

GPU:
- 行列積、畳み込み
- 自己注意計算
- 逆伝播

効果: GPUが常に計算できる状態を維持
```

---

## 実装のポイント

### 1. 勾配クリッピング（必須）

```python
# Transformerで重要
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 2. 位置エンコーディング（必須）

```python
class PositionalEncoding(nn.Module):
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]
```

### 3. 進捗表示（ユーザビリティ）

```python
print(f"\n最初のバッチデータ読み込み中（ワーカープロセス起動中）...")
print("※30秒〜1分かかる場合があります（Windowsのmultiprocessing起動）")
```

---

## 結論

| ポイント | 説明 |
|---------|------|
| **Transformer** | 1D信号処理に応用可能 |
| **セグメント分割** | 信号をトークンのように扱う |
| **埋め込み次元** | 256〜768（メモリに応じて調整） |
| **Batch Size** | 128〜256（RTX 3060推奨） |
| **num_workers** | CPUコア数に応じて2〜6 |
| **学習率** | 5e-4 + OneCycleLR |
| **GPU vs CPU** | GPUは計算、CPUはデータ読み込みで役割分担 |

**「データ数が多ければ、ResNetよりTransformerが有効」です。**
