# requirements.txt (Streamlit Cloud)
# streamlit
# transformers<5.0.0
# torch
# sentencepiece
# protobuf
# pandas
# numpy
# opencc-python-reimplemented

import pandas as pd
import numpy as np
import streamlit as st
from transformers import pipeline

# Optional: convert Simplified -> Traditional (HK)
try:
	from opencc import OpenCC
	_cc = OpenCC("s2hk")  # use "s2t" for generic Traditional
except Exception:
	_cc = None


# =========================
# Default models
# =========================
P1_MODEL_ID = "mingflam/final_sentiment_model_v3"
P2_MODEL_ID = "heack/HeackMT5-ZhSum100k"

LABEL_MAP = {
	"LABEL_0": "Positive",
	"LABEL_1": "Negative",
}


def get_device() -> int:
	# Streamlit Cloud is usually CPU; keep this for local GPU
	try:
		import torch
		return 0 if torch.cuda.is_available() else -1
	except Exception:
		return -1


@st.cache_resource
def load_pipelines():
	device = get_device()
	clf = pipeline(
		"text-classification",
		model=P1_MODEL_ID,
		tokenizer=P1_MODEL_ID,
		return_all_scores=False,
		device=device,
	)
	sum_pipe = pipeline(
		"summarization",
		model=P2_MODEL_ID,
		tokenizer=P2_MODEL_ID,
		framework="pt",
		device=device,
	)
	return clf, sum_pipe


def normalize_text(title: str, text: str) -> str:
	title = (title or "").strip()
	text = (text or "").strip()
	return (title + " " + text).strip()


def map_sentiment_label(raw_label: str) -> str:
	return LABEL_MAP.get(raw_label, raw_label)


def build_digest(text: str, label: str, score: float) -> str:
	return f"新聞內容: {text}\n情緒分類: {label} (confidence={score:.4f})\n請總結重點:"


def run_p1_batch(clf, texts, batch_size=16, max_length=512):
	outs = []
	for i in range(0, len(texts), batch_size):
		batch = texts[i : i + batch_size]
		out = clf(batch, truncation=True, padding=True, max_length=max_length)
		outs.extend(out)
	return outs


def run_p2(sum_pipe, digests, max_sum_len=80, min_sum_len=20):
	summaries = []
	for d in digests:
		d = (d or "").strip()[:4000]
		r = sum_pipe(
			d,
			max_length=max_sum_len,
			min_length=min_sum_len,
			truncation=True,
		)
		summary_text = r[0].get("summary_text", "")
		if _cc is not None:
			summary_text = _cc.convert(summary_text)
		summaries.append(summary_text)
	return summaries


def run_pipeline2_over_df(
	df,
	clf,
	sum_pipe,
	text_col="input_text",
	batch_size_clf=16,
	max_sum_len=80,
	min_sum_len=20,
):
	texts = df[text_col].fillna("").astype(str).tolist()

	# 1) Pipeline 1 in batches
	preds = run_p1_batch(clf, texts, batch_size=batch_size_clf, max_length=512)

	pred_labels = []
	pred_scores = []
	for p in preds:
		raw_label = p.get("label", "")
		score = float(p.get("score", np.nan))
		label = map_sentiment_label(raw_label)
		pred_labels.append(label)
		pred_scores.append(score)

	# 2) Build summarization input (digest)
	digests = [build_digest(t, lab, sc) for t, lab, sc in zip(texts, pred_labels, pred_scores)]

	# 3) Pipeline 2
	summaries = run_p2(sum_pipe, digests, max_sum_len=max_sum_len, min_sum_len=min_sum_len)

	out_df = df.copy()
	out_df["sentiment"] = pred_labels
	out_df["confidence"] = pred_scores
	out_df["p2_input_digest"] = digests
	out_df["summary"] = summaries
	return out_df


st.set_page_config(
	page_title="Chinese Finance News Briefing",
	layout="wide",
	initial_sidebar_state="collapsed",
)

st.title("Chinese Finance News: Sentiment + Summary")
st.caption("Upload a CSV (title,text) or run a single input. Sentiment uses P1; summary uses P2.")

clf, sum_pipe = load_pipelines()

tab1, tab2 = st.tabs(["Single input", "Batch CSV"])

with tab1:
	st.subheader("Single headline")
	col1, col2 = st.columns(2)
	with col1:
		title = st.text_input("title", value="")
	with col2:
		text = st.text_input("text", value="")
	input_text = normalize_text(title, text)

	if st.button("Run", type="primary"):
		if not input_text:
			st.warning("Please enter title/text.")
		else:
			with st.spinner("Running pipelines..."):
				df1 = pd.DataFrame({"input_text": [input_text]})
				out_df = run_pipeline2_over_df(df1, clf, sum_pipe, text_col="input_text")

			row = out_df.iloc[0]
			st.markdown("### Output")
			with st.expander("Input text (click to expand)", expanded=False):
				st.write(input_text)
			st.write("Sentiment:", row["sentiment"])
			st.write("Summary (Traditional Chinese HK):")
			st.write(row["summary"])

with tab2:
	st.subheader("Upload CSV → sentiment + summary + download")
	st.write("CSV must contain columns: title, text")
	uploaded = st.file_uploader("Upload CSV", type=["csv"])
	max_rows = int(st.number_input("Max rows (demo)", 10, 500, 100, 10))

	if uploaded is not None:
		df = pd.read_csv(uploaded)
		missing = [c for c in ["title", "text"] if c not in df.columns]
		if missing:
			st.error(f"Missing columns: {missing}")
		else:
			df = df.head(max_rows).copy()
			df["title"] = df["title"].fillna("").astype(str)
			df["text"] = df["text"].fillna("").astype(str)
			df["input_text"] = df.apply(lambda r: normalize_text(r["title"], r["text"]), axis=1)

			if st.button("Run batch", type="primary"):
				with st.spinner("Running pipelines (this can take a while on Streamlit Cloud CPU)..."):
					out_df = run_pipeline2_over_df(df, clf, sum_pipe, text_col="input_text")

				show_df = out_df[["sentiment", "confidence", "input_text", "p2_input_digest", "summary"]]
				st.dataframe(show_df.head(20), use_container_width=True)

				csv_bytes = show_df.to_csv(index=False).encode("utf-8")
				st.download_button(
					"Download results CSV",
					data=csv_bytes,
					file_name="pipeline2_results.csv",
					mime="text/csv",
				)
