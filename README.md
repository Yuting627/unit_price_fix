# 省份 × 品类份额异常

对当期每个省份的 category mix 并行打四种标记：`outlier_n0`、`outlier_n1_5`、`outlier_n6_11`、`outlier_n8`。标签为 `high` / `low` / `ok` / `na`。

## 份额定义

- **本省 share** = 该品类销售额 / 该省当期全部品类销售额
- **全国 share** = 该品类全国销售额 / 全国当期全部品类销售额
- **全国 share（leave-one-out）** = （全国该品类 − 本省该品类）/（全国合计 − 本省合计）
- **province_weight** = 该省销售额 / 全国当期销售额
- **n_hist** = 该省该品类在当期之前的历史期数（不含当期）
- **CLR / clr 近邻**：公式与例子见下一节。**n0 与 n1-5 共用**这套同伴。

四种方法都看份额，不看销售额横截面。识别函数在 `code/core/classify.py`：`identify_n0`、`identify_n1_5`、`identify_n6_11`、`identify_n8`。`main.py` 对每个省份×品类分别调用这四个函数并汇总。

## CLR 与 clr 近邻

CLR（centered log-ratio，中心化对数比）用来比较各省的品类 mix（份额加总为 1）。实现：`code/core/composition.py` 的 `clr_coords`、`aitchison_distance`、`nearest_peer_names`。

一省当期各品类份额 \(s_1,\ldots,s_D\)，且 \(\sum_i s_i=1\)。零份额先加 \(\varepsilon=10^{-6}\)，再归一化：

\[
\tilde s_i=\frac{s_i+\varepsilon}{\sum_j(s_j+\varepsilon)}
\]

CLR 坐标：对各分量取对数，再减去对数均值。

\[
\operatorname{clr}(\tilde s)_i=\log\tilde s_i-\frac{1}{D}\sum_{j=1}^{D}\log\tilde s_j
\]

Aitchison 距离是两条 CLR 向量的欧氏距离：

\[
d(p,q)=\bigl\|\operatorname{clr}(\tilde s^{(p)})-\operatorname{clr}(\tilde s^{(q)})\bigr\|_2
\]

检验品类 \(c\) 时做 **leave-one-category-out**：距离里不用 \(s_c\)，只用其余品类。对其他每个省算 \(d\)，由小到大取最近 8 个；有效同伴不足 6 个则扩到 12，还不够就用全部其他省。

**手算小例子**（丢掉啤酒后只剩水、粮，先各自归一化）：

- 广东：水 30%、粮 20% → \((0.6,\,0.4)\)，\(\operatorname{clr}\approx(0.203,\,-0.203)\)
- 江苏：水 32%、粮 20% → \((0.615,\,0.385)\)，\(\operatorname{clr}\approx(0.235,\,-0.235)\)，\(d\approx 0.046\)
- 西藏：水 10%、粮 85%，相对广东 \(d\approx 1.80\)

江苏进近邻，西藏不进。比的是其余品类结构像不像，不是省份大小，也不是啤酒份额本身。

**July 真数（广东 × BEER）**：广东啤酒份额约 1.18%。丢掉啤酒后最近 8 省：

| 近邻 | 该省啤酒 share | \(d\)（其余品类） |
|---|---|---|
| 广西 | 1.07% | 4.68 |
| 浙江 | 3.36% | 6.91 |
| 上海 | 3.86% | 7.58 |
| 湖南 | 6.99% | 8.53 |
| 云南 | 4.16% | 9.28 |
| 湖北 | 2.57% | 9.36 |
| 河南 | 4.35% | 9.96 |
| 山东 | 9.19% | 10.03 |

最远如福建、重庆、天津（\(d\) 约 22～28）。维数高时绝对距离会到十几，只按排序取最近 8 个。n0 用这 8 省的当期 share 做 Tukey；n1-5 用这 8 省的 \(\log(1+g)\) / Δpp 做 Tukey。

## n0：当期份额（不用历史）

函数：`identify_n0`

同伴：见上文 **clr 近邻**。用这些省在该品类上的当期 share 做 Tukey。同伴不足 4 个则为 `na`。

要标异常，须同时满足：

1. 本省 share 相对 clr 近邻是 Tukey 异常（1.5×IQR；IQR=0 时，不等于中位数即异常），且 `|本省 share − 同伴中位数| ≥ 1pp`
2. 本省 share ≥ 0.5%（更小品类直接降为 `ok`）
3. province_weight ≥ 1%（过小省份降为 `ok`）
4. 相对 **leave-one-out 全国份额** 同向，且 `|本省 share − 全国_{-p}| > 1pp`

全部品类打完后做 **整省封顶**：一省当期 n0 异常（high/low）≥ 5 个时，只按 `|本省 share − 全国_{-p}|` 保留最大的 2 个，其余降为 `ok`（`note` 记 `n0_province_cap`）。异常不足 5 个的省不改。

## n1-5：同伴份额变化（`n_hist ≥ 1`）

函数：`identify_n1_5`

需要至少 1 期历史，否则 `na`。不做整省封顶。自身 IQR 不用。**不含 Holt**。长历史键同样打这一列，用近 3 期均值作自身方向，再和 **clr 近邻**（与 n0 同一批省）比变化，不用全省一锅炖。

1. `|当期 − 近 min(3, n_hist) 期均值| > 0.01%`，定方向
2. **gr peer** 与 **Δpp peer** 取**交集**：本省 `log(1+g)` 和 Δpp 相对 clr 近邻都是 1.5×IQR 异常，且方向与自身一致
3. 任一条有效同伴不足 4 个则跳过，标 `ok`
4. 份额 < 0.5% 降为 `ok`
5. province_weight ≥ 1%（过小省份降为 `ok`，`note` 记 `n15_small_province`）

## n6-11：自身历史 Tukey（`n_hist ≥ 6`）

函数：`identify_n6_11`

不足 6 期则为 `na`。不做整省封顶，不用同伴变化。自身 Tukey **交集**（两条都有时必须同时过，方向一致）：

1. **对上一期**：`log(1+环比)` 或 Δpp（当期 − 上一期）相对本省历史一期变化是 1.5×IQR 异常。这条还要 `|当期 − 近 3 期均值| > 0.01%`，方向一致。
2. **对近 7 期**：当期相对最近 7 期均值的 Δpp 或 `log(1+g)`，对照本省滚动 7 期残差做 1.5×IQR，且 `|当期 − 7 期均值| ≥ 0.5pp`（有效残差不足 4 个则跳过，通常要 `n_hist ≥ 11`）。

一条跳过时只要求另一条。份额 < 0.5% 仍降为 `ok`。Holt 只在 n8。

## n8：Holt 趋势（n_hist ≥ 8）

函数：`identify_n8`

历史不足 8 期则为 `na`。只用该省该品类自己的历史 share（日历补齐），不含当期。

- 阻尼 Holt：\(\alpha=0.2,\ \beta=0.1,\ \phi=0.8\)
- 初值：level = 第 1 期，trend = 第 2 期 − 第 1 期
- 误差从第 3 个历史点起算，用 t 分布 99% 容差带
- 当期 share 落在带外则为 `high` / `low`；常数序列（\(\sigma=0\)）不标
- 本省 share（或上期 share）< 0.5% 时降为 `ok`

## 怎么跑

```bash
python main.py
```

读 `data/province_cat_salesvalue.xlsx`，写 `data/province_cat_salesvalue_outlier.xlsx`（detail + summary，含 n0 / n1-5 / n6-11 与 n8 的重叠）。
