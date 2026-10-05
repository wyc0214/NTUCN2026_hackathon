# PartGuard — 小型工廠的離線外觀品檢原型

在**裝置端**（不上雲）替金屬小零件（墊片、螺帽、沖壓環）做出貨前的最後一道外觀檢查，每個零件給出合格／不合格與原因：

| 檢查項目 | 原因代碼 |
|---|---|
| 外徑尺寸（對批次中位數或圖面公差） | `SIZE_SMALL` / `SIZE_LARGE` |
| 缺角、毛邊、變形（實心度、橢圓度） | `DEFORMED` |
| 孔洞：缺孔、孔徑異常、孔位偏心 | `HOLE_MISSING` / `HOLE_SIZE` / `HOLE_OFFCENTER` |
| 表面：銹斑（色相）、刮痕（細線偵測） | `RUST` / `SCRATCH` |

目前是**筆電可跑的原型**（OpenCV + NumPy）。同一套引擎也保留了食品範例（麵包、便當托盤），見 `partguard/bakery.py`、`partguard/bento.py`。

## 快速開始

```bash
pip install -r requirements.txt
python tools/make_parts_synthetic.py       # 產生「合成」金屬墊片測試圖（是繪製的，不是真實金屬）
python tools/make_synthetic.py             # （選用）食品範例的合成測試圖與參考圖庫
python tests/test_smoke.py                 # 煙霧測試
python run.py --mode parts --source samples/parts_batch_mixed.png --lang en
python run.py --mode parts --source 0 --show      # 接網路攝影機，按 q 離開
```

輸出：文字說明（`--lang zh|en`）、標註圖（`out/`）、逐次 JSON 紀錄與延遲（`logs/inspections.jsonl`）。

## 純電腦就能產出的結果

```bash
python tools/benchmark_parts.py     # 壓力測試 + 瑕疵嚴重度掃描，含 95% 信賴區間
python tools/roi_model.py           # 批次退貨損益平衡（所有輸入都是假設）
```

- `benchmark_parts.py` 比較「沒有灰卡／有灰卡／有灰卡且曝光預留餘裕」在亮度、偏色、雜訊下的誤報，並掃描瑕疵嚴重度 0.25–1.0，找出偵測在哪裡開始失效。
  **這是合成資料**，只能說明方法的敏感度，不能代表真實金屬零件（真實零件有反光、油污、毛邊、加工紋理）。
- `roi_model.py`：「系統一年要攔下幾批不良品才回本？」所有輸入（系統成本、單批退貨損失、攔截率）都是假設，請換成有來源的數字。

## 灰卡校正

在鏡頭畫面角落固定放一張灰卡（位置在 `configs/parts.yaml` 的 `calibration.card_roi`），程式依它把每張影像的亮度與色偏拉回基準。
**若不放灰卡，請刪掉 `calibration` 區塊**，否則會用錯誤的區域校色。校正補不回「過曝」：亮部一旦被截斷資訊就消失了，所以實際使用要讓相機曝光預留餘裕。

## 換成你們的真實零件

1. 固定相機、治具與光源（建議漫射光；金屬反光是最大的變數，可加偏光片）。
2. 拍 50 個以上你們認定的良品，量出 `px_per_mm`，把圖面公差填進 `configs/parts.yaml`。
3. 用良品的分布調整門檻（目前的門檻是依合成良品的分布擬合的）。
4. 另外收集真正的不良品，驗證抓得到。

## 已知限制（請在簡報中誠實呈現）

- 所有數字來自合成圖；真實金屬的反光、油污與加工紋理尚未驗證。
- 規則式方法對很輕微的瑕疵會漏檢（見嚴重度掃描）。Stage II 的方向是把學習式缺陷分類器放到 UGen300。
- `portable to UGen300`：`partguard/backends.py` 的 `HailoEmbedder` 仍是預留位置，實際支援的模型與工具以競賽「運算平台說明」頁為準。
- 公開的工業瑕疵資料集多為非商業授權，與競賽規則和商業用途可能衝突，使用前請逐一確認，本專案不內附任何第三方影像。
- 延遲數字是一般 CPU 的結果，不代表 UGen300。
