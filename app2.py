# import part
import pandas as pd
import numpy as np
import streamlit as st
from transformers import pipeline

# convert Simplified -> Traditional (HK)
try:
	from opencc import OpenCC
	_cc = OpenCC("s2hk")  # use "s2t" for generic Traditional
except Exception:
	_cc = None


# =========================
# Default models (no UI selection)
# =========================
P1_MODEL_ID = "mingflam/final_sentiment_model_v2"
P2_MODEL_ID = "heack/HeackMT5-ZhSum100k"

# Adjust mapping to your confirmed label meaning
LABEL_MAP = {
	"LABEL_0": "Positive",
	"LABEL_1": "Negative",
}


# function part
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
		device=device,
	)
	sum_pipe = pipeline(
		"summarization",
		model=P2_MODEL_ID,
		tokenizer=P2_MODEL_ID,
		device=device,
	)
	return clf, sum_pipe


def normalize_text(title: str, text: str) -> str:
	title = (title or "").strip()
	text = (text or "").strip()
	return (title + " " + text).strip()


def map_sentiment_label(raw_label: str) -> str:
	return LABEL_MAP.get(raw_label, raw_label)


def run_p1_batch(clf, texts, batch_size=16, max_length=512):
	outs = []
	for i in range(0, len(texts), batch_size):
		batch = texts[i : i + batch_size]
		out = clf(batch, truncation=True, padding=True, max_length=max_length)
		outs.extend(out)
	return outs


def run_p2_single(sum_pipe, text: str, max_len=120, min_len=30) -> str:
	text = (text or "").strip()
	if not text:
		return ""
	text = text[:4000]
	res = sum_pipe(text, max_length=max_len, min_length=min_len, truncation=True)
	summary_text = res[0].get("summary_text", "")
	# Convert Simplified -> Traditional (HK) if OpenCC is available
	if _cc is not None:
		summary_text = _cc.convert(summary_text)
	return summary_text


def output_msg(input_text: str, sentiment_label: str, summary_text: str):
	st.markdown("### Output")
	with st.expander("Input text (click to expand)", expanded=False):
		st.write(input_text)
	st.write("Sentiment:", sentiment_label)
	st.write("Summary (Pipeline 2):")
	st.write(summary_text)


# main part
st.set_page_config(
	page_title="Chinese Finance News Briefing",
	layout="wide",
	initial_sidebar_state="collapsed",
)

st.title("Chinese Finance News: Sentiment + Summary")
st.caption("Pipeline 1 = sentiment classifier. Pipeline 2 = summarization.")

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
			with st.spinner("Running sentiment..."):
				out = run_p1_batch(clf, [input_text], batch_size=1)
				p = out[0]
				raw = p.get("label", "")
				sentiment = map_sentiment_label(raw)

			with st.spinner("Running summarization (this can take a while on Streamlit Cloud CPU)..."):
				summary = run_p2_single(sum_pipe, input_text)
			output_msg(input_text, sentiment, summary)

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
				with st.spinner("Running sentiment..."):
					texts = df["input_text"].tolist()
					preds = run_p1_batch(clf, texts, batch_size=16, max_length=512)
					sentiments = [map_sentiment_label(p.get("label", "")) for p in preds]

				with st.spinner("Running summarization (this can take a while on Streamlit Cloud CPU)..."):
					summaries = [run_p2_single(sum_pipe, t) for t in df["input_text"].tolist()]

				out_df = pd.DataFrame({
					"sentiment": sentiments,
					"input_text": df["input_text"].tolist(),
					"summary": summaries,
				})

				st.dataframe(out_df.head(20), use_container_width=True)

				csv_bytes = out_df.to_csv(index=False).encode("utf-8")
				st.download_button(
					"Download results CSV",
					data=csv_bytes,
					file_name="sentiment_results.csv",
					mime="text/csv",
				)
