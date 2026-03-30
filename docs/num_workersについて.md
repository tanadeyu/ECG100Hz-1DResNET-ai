# num_workersについて

## 要点

- **num_workers**: データ読み込みの並列化数
- **Windows**: `0`推奨（spawn方式の問題）
- **Linux**: `4〜8`推奨（fork方式で高速）
- **0でも問題ない**: RTX 3060なら十分高速

---

## num_workersとは

### 定義

**DataLoaderでデータを並列読み込みするためのワーカー（プロセス）数**

### 役割

```
num_workers=0:
メインプロセスがデータを読み込む（シリアル）

num_workers=4:
4つの子プロセスが並列でデータを読み込む（パラレル）
```

---

## OSによる違い

### Linux（fork方式）

```
親プロセス [データA, B, C, D]
    ↓ コピーなしでメモリ共有
子プロセス1 → [データA] を処理
子プロセス2 → [データB] を処理
子プロセス3 → [データC] を処理
子プロセス4 → [データD] を処理
```

**メリット**:
- メモリ共有で高速
- プロセス起動が早い
- 安定して動作

### Windows（spawn方式）

```
親プロセス [データA, B, C, D]
    ↓ 全コピーしてから分割
子プロセス1 → [データA,B,C,Dのコピー] からAを処理
子プロセス2 → [データA,B,C,Dのコピー] からBを処理
子プロセス3 → [データA,B,C,Dのコピー] からCを処理
子プロセス4 → [データA,B,C,Dのコピー] からDを処理
```

**デメリット**:
- メモリ使用量が倍増
- プロセス起動に時間がかかる
- デッドロックの可能性

---

## 推奨設定

| OS | 推奨num_workers | 理由 |
|----|-----------------|------|
| **Windows** | 0 | spawn方式の問題回避 |
| **Linux** | 4〜8 | fork方式で高速化 |
| **macOS** | 0〜2 | macOSもspawn方式 |

### Windowsでの詳細

| num_workers | 動作 | 問題 |
|-------------|------|------|
| **0** | 安定 | なし（推奨） |
| 1〜2 | 動作する場合あり | デッドロックの可能性 |
| 3以上 | 問題発生の可能性大 | プロセス起動で数分以上停止 |

---

## 実際の問題

### 発生した問題

```
設定: num_workers=4
結果: 3分待っても進まない

原因:
- 4つの子プロセスを作成
- 各プロセスがデータセット全体（17,441件）をコピー
- メモリ不足またはデッドロック発生
```

### 対処法

```python
# 修正前
NUM_WORKERS = 4  # Windowsでは問題発生

# 修正後
NUM_WORKERS = 0  # Windowsで安定動作
```

---

## 速度比較

### 理論上の速度

| num_workers | GPU利用率 | 1Epoch時間 | 備考 |
|-------------|----------|-----------|------|
| 0 | 85〜95% | 5〜6秒 | Windows推奨 |
| 2 | 90〜95% | 4〜5秒 | Linuxのみ |
| 4 | 95〜100% | 4秒以下 | Linuxのみ |

### 実測（RTX 3060、Windows）

| num_workers | 1Epoch時間 | メモリ使用量 |
|-------------|-----------|-------------|
| 0 | 5〜6秒 | 正常 |
| 2 | 4〜5秒 | 増加 |
| 4 | **進まない** | 過大 |

**Windowsではnum_workersを増やしても、劇的な速度向上は期待できません。**

---

## PyTorch DataLoaderの設定

### 推奨コード（Windows）

```python
from torch.utils.data import DataLoader

# Windows推奨設定
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,      # Windowsでは0
    pin_memory=True     # GPU転送最適化
)
```

### Linuxの場合

```python
# Linux推奨設定
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,      # Linuxでは4〜8
    pin_memory=True
)
```

---

## pin_memoryとの組み合わせ

```python
DataLoader(
    ...,
    num_workers=0,
    pin_memory=True  # GPU転送を高速化
)
```

### pin_memoryの効果

| 設定 | メモリ領域 | GPU転送速度 |
|------|-----------|-----------|
| pin_memory=False | 通常メモリ | 通常 |
| pin_memory=True | ページロックメモリ | 高速 |

**num_workers=0 + pin_memory=True が、Windowsでの最適設定です。**

---

## 結論

| ポイント | 説明 |
|---------|------|
| **Windows** | num_workers=0推奨 |
| **Linux** | num_workers=4〜8推奨 |
| **速度差** | Windowsではあまり変わらない |
| **安定性** | 0が最安定 |
| **設定** | `num_workers=0, pin_memory=True` |

**「とりあえず0にしておけば問題ない」です。**
