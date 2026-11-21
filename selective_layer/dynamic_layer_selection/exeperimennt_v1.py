import json
import math
import os
from contextlib import contextmanager
from functools import reduce
from math import gcd
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from rouge_score import rouge_scorer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==============================================================================
# 1. 設定 (CFG)
# ==============================================================================
CFG = {
    "model_name": "facebook/opt-iml-1.3b",
    "dataset_name": "tatsu-lab/alpaca",
    "n_samples": 500,          # 評価プロンプト数（論文は 7800）
    "n_train": 2000,           # LoRA 学習件数（論文はフル微調整で全量）
    "max_prompt_tokens": 256,
    "max_new_tokens": 128,
    "max_seq_len": 512,        # 教師あり微調整(SFT)時の最大系列長
    "batch_size": 16,
    "costs": [4, 8, 12, 24],
    "seed": 0,
    "out_dir": "./seq_oracle_out",
    "ft_epochs": 1,
    "ft_lr": 2e-4,
    "ft_batch": 8,
    "ft_accum": 2,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "router_layer": 12,        # OPT の第12層隠れ状態を特徴量に利用
    "cv_splits": 5,
    "num_random_mix_trials": 20,
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 2. データ準備（重複防止のため評価は末尾15%、学習は先頭70%から抽出）
# ==============================================================================
def format_alpaca_prompt(example: Dict[str, str]) -> str:
    instruction = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    if inp:
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            "### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )


def prepare_alpaca_splits(
    dataset_name: str,
    n_train: int,
    n_eval: int,
    seed: int = 0,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    ds = load_dataset(dataset_name, split="train")
    total = len(ds)
    rng = np.random.default_rng(seed)

    # 先頭70%から学習用を抽出、末尾15%から評価用を抽出（分割重複を防止）
    train_pool_end = int(total * 0.70)
    eval_pool_start = int(total * 0.85)

    train_pool = rng.permutation(np.arange(0, train_pool_end))[: min(n_train, train_pool_end)]
    eval_pool = rng.permutation(np.arange(eval_pool_start, total))[: min(n_eval, total - eval_pool_start)]

    def _build_records(indices: np.ndarray) -> List[Dict[str, str]]:
        records = []
        for idx in indices:
            row = ds[int(idx)]
            records.append(
                {
                    "prompt": format_alpaca_prompt(row),
                    "response": (row.get("output") or "").strip(),
                }
            )
        return records

    return _build_records(train_pool), _build_records(eval_pool)


# ==============================================================================
# 3. Uniform Layer Skipping (ULS) と KV-Cache layer_idx 振り直し
# ==============================================================================
def uniform_pattern(num_layers: int, c: int) -> List[int]:
    """論文式(2)に従い、1層目を残しつつ深さ方向に均等に c 層を残す 0/1 リストを返す。"""
    assert 1 <= c <= num_layers, f"Invalid c={c} for num_layers={num_layers}"
    u = [1]
    for l in range(2, num_layers + 1):
        u.append(1 if sum(u) <= (l - 1) * c / num_layers + 1e-9 else 0)
    # 残す層数が c とずれるケース（丸め境界）を均等間隔で調整
    while sum(u) < c:
        for idx in range(num_layers - 1, 0, -1):
            if u[idx] == 0:
                u[idx] = 1
                break
    while sum(u) > c:
        for idx in range(num_layers - 1, 0, -1):
            if u[idx] == 1:
                u[idx] = 0
                break
    return u


def _get_opt_decoder(model: nn.Module) -> nn.Module:
    """Base OPT または PeftModel(OPT) の双方から decoder を取得する。"""
    base = model.get_base_model() if isinstance(model, PeftModel) else model
    if hasattr(base, "model") and hasattr(base.model, "decoder"):
        return base.model.decoder
    if hasattr(base, "decoder"):
        return base.decoder
    raise ValueError("Unsupported model structure: OPT decoder not found.")


@contextmanager
def skip_layers(model: nn.Module, c: int):
    """
    指定した層数 c だけ残してデコーダ層を一時的にスキップするコンテキストマネージャ。
    KVキャッシュのインデックス不整合を防ぐため、残す層の self_attn.layer_idx を 0..k-1 に振り直す。
    """
    dec = _get_opt_decoder(model)
    orig_layers = dec.layers
    orig_n = len(orig_layers)
    orig_cfg_layers = getattr(model.config, "num_hidden_layers", orig_n)

    keep_indices = [i for i, flag in enumerate(uniform_pattern(orig_n, c)) if flag == 1]

    saved_layer_idx = []
    for layer in orig_layers:
        attn = getattr(layer, "self_attn", None)
        saved_layer_idx.append((attn, getattr(attn, "layer_idx", None) if attn is not None else None))

    for new_pos, orig_idx in enumerate(keep_indices):
        attn = getattr(orig_layers[orig_idx], "self_attn", None)
        if attn is not None and hasattr(attn, "layer_idx"):
            attn.layer_idx = new_pos

    dec.layers = nn.ModuleList([orig_layers[i] for i in keep_indices])
    model.config.num_hidden_layers = len(keep_indices)

    try:
        yield keep_indices
    finally:
        dec.layers = orig_layers
        model.config.num_hidden_layers = orig_cfg_layers
        for attn, old_idx in saved_layer_idx:
            if attn is not None and old_idx is not None and hasattr(attn, "layer_idx"):
                attn.layer_idx = old_idx


# ==============================================================================
# 4. 生成・ROUGE-L 行列 R の算出とキャッシュ
# ==============================================================================
@torch.no_grad()
def generate_for_cost(
    model: nn.Module,
    tokenizer,
    eval_records: List[Dict[str, str]],
    c: int,
    tag: str,
    cfg: Dict,
    device: torch.device,
    run_idx: int = 0,
) -> List[str]:
    os.makedirs(cfg["out_dir"], exist_ok=True)
    suffix = f"_run{run_idx}" if run_idx > 0 else ""
    cache_path = os.path.join(cfg["out_dir"], f"gens{tag}_c{c}_n{len(eval_records)}{suffix}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    prompts = [r["prompt"] for r in eval_records]
    # 長さ順に並べてパディングオーバーヘッドを削減しつつ元の順序へ復元する
    order = np.argsort([len(p) for p in prompts])
    inv_order = np.argsort(order)
    sorted_prompts = [prompts[i] for i in order]

    model.eval()
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    preds_sorted: List[str] = []
    batch_size = cfg["batch_size"]

    with skip_layers(model, c):
        for start in range(0, len(sorted_prompts), batch_size):
            batch_prompts = sorted_prompts[start : start + batch_size]
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg["max_prompt_tokens"],
            ).to(device)

            input_len = enc["input_ids"].shape[1]
            # run_idx > 0（クロスラン検証）ではサンプリング生成、通常評価は greedy
            do_sample = run_idx > 0
            gen_ids = model.generate(
                **enc,
                max_new_tokens=cfg["max_new_tokens"],
                do_sample=do_sample,
                temperature=0.7 if do_sample else 1.0,
                top_p=0.9 if do_sample else 1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

            for row_ids in gen_ids:
                new_tokens = row_ids[input_len:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                preds_sorted.append(text)

    tokenizer.padding_side = orig_padding_side
    preds = [preds_sorted[i] for i in inv_order]

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)

    return preds


def build_rouge_matrix(
    preds_by_cost: Dict[int, List[str]],
    eval_records: List[Dict[str, str]],
    costs: List[int],
) -> np.ndarray:
    """形状 [N, K] の ROUGE-L F-measure 行列 R を作成する。"""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    n_samples = len(eval_records)
    num_costs = len(costs)
    r_mat = np.zeros((n_samples, num_costs), dtype=np.float64)

    refs = [rec["response"] for rec in eval_records]
    for k_idx, c in enumerate(costs):
        hyps = preds_by_cost[c]
        for i in range(n_samples):
            score = scorer.score(refs[i], hyps[i])["rougeL"].fmeasure
            r_mat[i, k_idx] = score
    return r_mat


# ==============================================================================
# 5. 多選択ナップサック DP オラクル / Greedy / Random Mix / クロスラン
# ==============================================================================
def solve_oracle_dp(r_mat: np.ndarray, costs: List[int], beta: float) -> Tuple[np.ndarray, float]:
    """
    各プロンプト i ∈ [0, N-1] に深さ k ∈ [0, K-1] を1つ割り当て、
    平均コスト (1/N) * sum(costs[k_i]) <= beta の下で平均 ROUGE-L を最大化する。
    コストを最大公約数 g で割って整数化し、DP で厳密解を求める。
    """
    n_samples, num_costs = r_mat.shape
    g = reduce(gcd, costs)
    u = [c // g for c in costs]

    max_avg_u = beta / g
    max_total_u = int(math.floor(max_avg_u * n_samples + 1e-9))
    max_total_u = max(max_total_u, min(u) * n_samples)

    neg_inf = -1e18
    # dp[w]: 現在のプロンプト数まで処理したときの総コストユニット w での最大スコア合計
    dp = np.full(max_total_u + 1, neg_inf, dtype=np.float64)
    dp[0] = 0.0

    # 逆再生用の選択記録: choices[i, w] = 選んだ cost index k
    choices = np.full((n_samples, max_total_u + 1), -1, dtype=np.int32)

    for i in range(n_samples):
        next_dp = np.full(max_total_u + 1, neg_inf, dtype=np.float64)
        row_scores = r_mat[i]
        for k_idx, cost_u in enumerate(u):
            sc = row_scores[k_idx]
            if cost_u > max_total_u:
                continue
            valid_prev = np.where(dp[: max_total_u - cost_u + 1] > neg_inf / 2)[0]
            if valid_prev.size == 0:
                continue
            cand_scores = dp[valid_prev] + sc
            target_w = valid_prev + cost_u
            for w_idx, w_val in enumerate(target_w):
                if cand_scores[w_idx] > next_dp[w_val]:
                    next_dp[w_val] = cand_scores[w_idx]
                    choices[i, w_val] = k_idx
        dp = next_dp

    best_w = int(np.argmax(dp))
    if dp[best_w] < neg_inf / 2:
        # 予算が極端に小さい場合は最小コストを割り当て
        selected_k = np.full(n_samples, int(np.argmin(costs)), dtype=np.int32)
    else:
        selected_k = np.zeros(n_samples, dtype=np.int32)
        cur_w = best_w
        for i in range(n_samples - 1, -1, -1):
            k_idx = int(choices[i, cur_w])
            if k_idx < 0:
                k_idx = int(np.argmin(costs))
            selected_k[i] = k_idx
            cur_w -= u[k_idx]

    mean_score = float(np.mean(r_mat[np.arange(n_samples), selected_k]))
    return selected_k, mean_score


def greedy_oracle_score(r_mat: np.ndarray, costs: List[int], beta: float) -> Tuple[float, float]:
    """
    各プロンプトで c <= beta を満たす深さのうち最高スコアを選ぶ（予算の系列間融通なし）。
    """
    n_samples, _ = r_mat.shape
    allowed_indices = [idx for idx, c in enumerate(costs) if c <= beta + 1e-9]
    if not allowed_indices:
        allowed_indices = [int(np.argmin(costs))]

    sub_r = r_mat[:, allowed_indices]
    best_local = np.argmax(sub_r, axis=1)
    chosen_k = np.array([allowed_indices[idx] for idx in best_local], dtype=np.int32)
    mean_score = float(np.mean(r_mat[np.arange(n_samples), chosen_k]))
    mean_cost = float(np.mean(np.array(costs)[chosen_k]))
    return mean_score, mean_cost


def random_mix_score(
    r_mat: np.ndarray,
    costs: List[int],
    selected_k: np.ndarray,
    num_trials: int = 20,
    seed: int = 0,
) -> float:
    """
    oracle と同じ深さ選択比率を保ったまま、プロンプトへランダム割り当てしたときの平均 ROUGE-L。
    """
    rng = np.random.default_rng(seed)
    n_samples, num_costs = r_mat.shape
    counts = np.bincount(selected_k, minlength=num_costs)
    probs = counts / max(1, counts.sum())

    trial_scores = []
    for _ in range(num_trials):
        rand_k = rng.choice(num_costs, size=n_samples, p=probs)
        trial_scores.append(float(np.mean(r_mat[np.arange(n_samples), rand_k])))
    return float(np.mean(trial_scores))


def analyze_oracle(
    r_mat: np.ndarray,
    costs: List[int],
    betas: List[float] = None,
    num_random_mix_trials: int = 20,
    seed: int = 0,
) -> Dict:
    """Part A / Part B 共通のオラクル・静的・greedy・random mix 分析関数。"""
    if betas is None:
        betas = [float(c) for c in range(min(costs), max(costs) + 1)]

    max_cost = max(costs)
    full_idx = costs.index(max_cost)
    full_score = float(np.mean(r_mat[:, full_idx]))

    static_scores = {
        int(c): float(np.mean(r_mat[:, idx]))
        for idx, c in enumerate(costs)
    }
    greedy_full_score, _ = greedy_oracle_score(r_mat, costs, float(max_cost))

    rows = []
    min_beta_matching_full = None

    for beta in betas:
        sel_k, oracle_sc = solve_oracle_dp(r_mat, costs, beta)
        actual_cost = float(np.mean(np.array(costs)[sel_k]))
        greedy_sc, _ = greedy_oracle_score(r_mat, costs, beta)
        rand_mix_sc = random_mix_score(r_mat, costs, sel_k, num_trials=num_random_mix_trials, seed=seed)

        ratios = {
            int(c): float(np.mean(sel_k == idx))
            for idx, c in enumerate(costs)
        }

        if min_beta_matching_full is None and oracle_sc >= full_score:
            min_beta_matching_full = {
                "beta_budget": beta,
                "actual_mean_layers": actual_cost,
                "ratio_to_full_layers": actual_cost / max_cost,
                "oracle_rougeL": oracle_sc,
                "selection_ratios": ratios,
            }

        rows.append(
            {
                "beta": beta,
                "actual_mean_layers": actual_cost,
                "oracle": oracle_sc,
                "greedy": greedy_sc,
                "random_mix": rand_mix_sc,
                "ratios": ratios,
            }
        )

    return {
        "full_score": full_score,
        "greedy_at_max_cost": greedy_full_score,
        "static_scores": static_scores,
        "min_beta_matching_full": min_beta_matching_full,
        "curve": rows,
    }


def cross_run_oracle_eval(
    r_assign: np.ndarray,
    r_eval: np.ndarray,
    costs: List[int],
    betas: List[float],
) -> List[Dict]:
    """
    クロスラン・オラクル：
    r_assign（1回目の生成）で DP 割当てを決定し、r_eval（2回目の独立生成）でスコアを評価する。
    出力ノイズへの事後的な当てはめ（sampling advantage）を排除して検証する。
    """
    n_samples = r_eval.shape[0]
    results = []
    for beta in betas:
        sel_k, _ = solve_oracle_dp(r_assign, costs, beta)
        eval_sc = float(np.mean(r_eval[np.arange(n_samples), sel_k]))
        actual_cost = float(np.mean(np.array(costs)[sel_k]))
        results.append(
            {
                "beta": beta,
                "actual_mean_layers": actual_cost,
                "cross_run_rougeL": eval_sc,
            }
        )
    return results


# ==============================================================================
# 6. 学習ルーター（第12層平均プーリング + log(len) -> Ridge回帰 + τスイープ）
# ==============================================================================
@torch.no_grad()
def extract_router_features(
    model: nn.Module,
    tokenizer,
    records: List[Dict[str, str]],
    layer_idx: int,
    max_prompt_tokens: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """フル層構成でプロンプトを通し、指定層（デフォルト12層）の mean-pooled 隠れ状態 + log(len) を抽出。"""
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    feats_all = []
    prompts = [r["prompt"] for r in records]

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        ).to(device)

        outputs = model(**enc, output_hidden_states=True, use_cache=False)
        # hidden_states[0] が embedding、hidden_states[layer_idx] が第 layer_idx 層出力
        safe_layer = min(layer_idx, len(outputs.hidden_states) - 1)
        hs = outputs.hidden_states[safe_layer].float()  # [B, S, D]

        mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, S, 1]
        lengths = mask.sum(dim=1).clamp(min=1.0)            # [B, 1]
        pooled = (hs * mask).sum(dim=1) / lengths           # [B, D]
        log_len = torch.log(lengths)                        # [B, 1]

        feat = torch.cat([pooled, log_len], dim=-1).cpu().numpy()
        feats_all.append(feat)

    return np.vstack(feats_all)


def analyze_router(
    features: np.ndarray,
    r_mat: np.ndarray,
    costs: List[int],
    cv_splits: int = 5,
    taus: List[float] = None,
    num_random_mix_trials: int = 20,
    seed: int = 0,
) -> Dict:
    """
    各深さ c の ROUGE-L を Ridge 回帰（5-fold CV の out-of-fold 予測）で予測し、
    「予測スコアが c=max より tau 以内に収まる最安の深さ」を選択する。
    """
    if taus is None:
        taus = [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 1.00]

    n_samples, num_costs = r_mat.shape
    full_idx = int(np.argmax(costs))
    oof_preds = np.zeros_like(r_mat, dtype=np.float64)
    correlations = {}

    kf = KFold(n_splits=min(cv_splits, n_samples), shuffle=True, random_state=seed)
    for k_idx, c in enumerate(costs):
        y = r_mat[:, k_idx]
        for train_idx, val_idx in kf.split(features):
            reg = Ridge(alpha=1.0)
            reg.fit(features[train_idx], y[train_idx])
            oof_preds[val_idx, k_idx] = reg.predict(features[val_idx])

        if np.std(y) > 1e-12 and np.std(oof_preds[:, k_idx]) > 1e-12:
            corr = float(np.corrcoef(y, oof_preds[:, k_idx])[0, 1])
        else:
            corr = 0.0
        correlations[int(c)] = corr

    # コスト昇順に並べて閾値 tau 判定を行う
    sorted_k = np.argsort(costs)
    router_curve = []

    for tau in taus:
        chosen_k = np.full(n_samples, full_idx, dtype=np.int32)
        for i in range(n_samples):
            target_pred = oof_preds[i, full_idx] - tau
            for k_idx in sorted_k:
                if oof_preds[i, k_idx] >= target_pred:
                    chosen_k[i] = k_idx
                    break

        actual_rouge = float(np.mean(r_mat[np.arange(n_samples), chosen_k]))
        actual_cost = float(np.mean(np.array(costs)[chosen_k]))
        rand_mix_sc = random_mix_score(
            r_mat, costs, chosen_k, num_trials=num_random_mix_trials, seed=seed
        )
        router_curve.append(
            {
                "tau": tau,
                "actual_mean_layers": actual_cost,
                "router_rougeL": actual_rouge,
                "random_mix_rougeL": rand_mix_sc,
                "gain_over_random_mix": actual_rouge - rand_mix_sc,
            }
        )

    return {
        "correlations": correlations,
        "curve": router_curve,
    }


# ==============================================================================
# 7. Part B：深さ別 LoRA 微調整（応答部分のみ損失計算）
# ==============================================================================
class AlpacaSFTDataset(Dataset):
    def __init__(self, records: List[Dict[str, str]], tokenizer, max_len: int = 512):
        self.samples = []
        eos_token = tokenizer.eos_token or ""
        for rec in records:
            prompt_ids = tokenizer(rec["prompt"], add_special_tokens=True)["input_ids"]
            resp_ids = tokenizer(rec["response"] + eos_token, add_special_tokens=False)["input_ids"]
            if len(resp_ids) == 0:
                continue

            input_ids = (prompt_ids + resp_ids)[:max_len]
            prompt_len = min(len(prompt_ids), len(input_ids))
            labels = ([-100] * prompt_len) + input_ids[prompt_len:]

            if any(lbl != -100 for lbl in labels):
                self.samples.append(
                    {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def make_collate_fn(pad_token_id: int):
    def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].size(0) for item in batch)
        bsz = len(batch)
        input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        labels = torch.full((bsz, max_len), -100, dtype=torch.long)

        for i, item in enumerate(batch):
            l = item["input_ids"].size(0)
            input_ids[i, :l] = item["input_ids"]
            attention_mask[i, :l] = 1
            labels[i, :l] = item["labels"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return _collate


def train_lora_for_cost(
    peft_model: PeftModel,
    adapter_name: str,
    c: int,
    train_records: List[Dict[str, str]],
    tokenizer,
    cfg: Dict,
    device: torch.device,
) -> None:
    """指定された深さ c のスキップ構成のまま、応答部分のみに損失をかけて LoRA を学習する。"""
    peft_model.set_adapter(adapter_name)
    peft_model.train()

    dataset = AlpacaSFTDataset(train_records, tokenizer, max_len=cfg["max_seq_len"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["ft_batch"],
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
    )

    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg["ft_lr"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    optimizer.zero_grad()
    step = 0

    with skip_layers(peft_model, c):
        for epoch in range(cfg["ft_epochs"]):
            running_loss = 0.0
            for batch in loader:
                step += 1
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    outputs = peft_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                    )
                    loss = outputs.loss / cfg["ft_accum"]

                scaler.scale(loss).backward()
                running_loss += outputs.loss.item()

                if step % cfg["ft_accum"] == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            if step % cfg["ft_accum"] != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            avg_loss = running_loss / max(1, len(loader))
            print(f"[LoRA c={c:2d} | adapter={adapter_name}] epoch={epoch + 1}/{cfg['ft_epochs']} loss={avg_loss:.4f}")


# ==============================================================================
# 8. 結果表示ヘルパー
# ==============================================================================
def print_analysis_summary(title: str, res: Dict, costs: List[int]) -> None:
    print(f"\n==================== {title} ====================")
    print(f"Static ROUGE-L by cost: {res['static_scores']}")
    print(f"Full ({max(costs)} layers) ROUGE-L: {res['full_score']:.4f} | Greedy@max: {res['greedy_at_max_cost']:.4f}")
    m = res["min_beta_matching_full"]
    if m is not None:
        print(
            f"Min beta matching full: budget={m['beta_budget']:.1f}, "
            f"actual_layers={m['actual_mean_layers']:.2f} ({m['ratio_to_full_layers']*100:.1f}%), "
            f"oracle={m['oracle_rougeL']:.4f}, ratios={m['selection_ratios']}"
        )
    else:
        print("Min beta matching full: not reached within tested betas.")

    print("\n[Budget Curve]")
    for row in res["curve"]:
        r_str = ", ".join([f"c={c}:{row['ratios'][c]:.3f}" for c in costs])
        print(
            f"beta={row['beta']:4.1f} (mean={row['actual_mean_layers']:4.1f}) | "
            f"oracle={row['oracle']:.4f} | greedy={row['greedy']:.4f} | "
            f"rand_mix={row['random_mix']:.4f} | {r_str}"
        )


def print_router_summary(title: str, rt: Dict) -> None:
    print(f"\n-------------------- {title} --------------------")
    print(f"OOF Correlation with ROUGE-L: {rt['correlations']}")
    for row in rt["curve"]:
        print(
            f"tau={row['tau']:.2f} (mean={row['actual_mean_layers']:4.1f}) | "
            f"router={row['router_rougeL']:.4f} | rand_mix={row['random_mix_rougeL']:.4f} | "
            f"diff={row['gain_over_random_mix']:+.4f}"
        )


# ==============================================================================
# 9. メイン実験パイプライン（Part A -> Part B -> Cross-Run）
# ==============================================================================
def main(cfg: Dict = CFG) -> None:
    set_seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_records, eval_records = prepare_alpaca_splits(
        cfg["dataset_name"],
        n_train=cfg["n_train"],
        n_eval=cfg["n_samples"],
        seed=cfg["seed"],
    )

    # ---- Part A：微調整なしの素のモデル ----
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=dtype).to(device)

    preds0 = {}
    for c in cfg["costs"]:
        preds0[c] = generate_for_cost(base_model, tokenizer, eval_records, c, tag="_base", cfg=cfg, device=device)

    r0_mat = build_rouge_matrix(preds0, eval_records, cfg["costs"])
    res0 = analyze_oracle(
        r0_mat,
        cfg["costs"],
        num_random_mix_trials=cfg["num_random_mix_trials"],
        seed=cfg["seed"],
    )
    print_analysis_summary("Part A: Base (Pre-FT) Oracle Analysis", res0, cfg["costs"])

    feats0 = extract_router_features(
        base_model,
        tokenizer,
        eval_records,
        layer_idx=cfg["router_layer"],
        max_prompt_tokens=cfg["max_prompt_tokens"],
        batch_size=cfg["batch_size"],
        device=device,
    )
    rt0 = analyze_router(
        feats0,
        r0_mat,
        cfg["costs"],
        cv_splits=cfg["cv_splits"],
        num_random_mix_trials=cfg["num_random_mix_trials"],
        seed=cfg["seed"],
    )
    print_router_summary("Part A: Base Ridge Router", rt0)

    # ---- Part B：深さ c ごとの LoRA 微調整 ----
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    )

    first_adapter = f"c{cfg['costs'][0]}"
    peft_model = get_peft_model(base_model, lora_config, adapter_name=first_adapter)
    for c in cfg["costs"][1:]:
        peft_model.add_adapter(f"c{c}", lora_config)

    for c in cfg["costs"]:
        adapter_name = f"c{c}"
        train_lora_for_cost(peft_model, adapter_name, c, train_records, tokenizer, cfg, device)
        ckpt_path = os.path.join(
            cfg["out_dir"], f"lora_c{c}_n{len(train_records)}_e{cfg['ft_epochs']}_{adapter_name}"
        )
        peft_model.save_pretrained(ckpt_path, selected_adapters=[adapter_name])

    preds_ft = {}
    for c in cfg["costs"]:
        peft_model.set_adapter(f"c{c}")
        preds_ft[c] = generate_for_cost(peft_model, tokenizer, eval_records, c, tag="_ft", cfg=cfg, device=device)

    rf_mat = build_rouge_matrix(preds_ft, eval_records, cfg["costs"])
    res_f = analyze_oracle(
        rf_mat,
        cfg["costs"],
        num_random_mix_trials=cfg["num_random_mix_trials"],
        seed=cfg["seed"],
    )
    print_analysis_summary("Part B: Post-LoRA Oracle Analysis", res_f, cfg["costs"])

    # 微調整後のルーター評価（ベースモデルから特徴量抽出）
    peft_model.disable_adapter_layers()
    feats_f = extract_router_features(
        peft_model,
        tokenizer,
        eval_records,
        layer_idx=cfg["router_layer"],
        max_prompt_tokens=cfg["max_prompt_tokens"],
        batch_size=cfg["batch_size"],
        device=device,
    )
    peft_model.enable_adapter_layers()

    rt_f = analyze_router(
        feats_f,
        rf_mat,
        cfg["costs"],
        cv_splits=cfg["cv_splits"],
        num_random_mix_trials=cfg["num_random_mix_trials"],
        seed=cfg["seed"],
    )
    print_router_summary("Part B: Post-LoRA Ridge Router", rt_f)

    # ---- 追加実験：クロスラン・オラクル（サンプリング生成2回目で評価） ----
    preds_ft_run2 = {}
    for c in cfg["costs"]:
        peft_model.set_adapter(f"c{c}")
        preds_ft_run2[c] = generate_for_cost(
            peft_model,
            tokenizer,
            eval_records,
            c,
            tag="_ft",
            cfg=cfg,
            device=device,
            run_idx=2,
        )
    rf_mat_run2 = build_rouge_matrix(preds_ft_run2, eval_records, cfg["costs"])
    key_betas = [4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
    xrun_res = cross_run_oracle_eval(rf_mat, rf_mat_run2, cfg["costs"], key_betas)

    print("\n-------------------- Cross-Run Oracle (Assign Run1 -> Eval Run2) --------------------")
    for row in xrun_res:
        print(
            f"beta={row['beta']:4.1f} (mean={row['actual_mean_layers']:4.1f}) | "
            f"cross_run_rougeL={row['cross_run_rougeL']:.4f}"
        )


if __name__ == "__main__":
    main()