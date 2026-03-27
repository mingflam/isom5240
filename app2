# install part
# requirements.txt (Streamlit Cloud)
# streamlit
# transformers<5.0.0
# torch
# sentencepiece
# protobuf
# opencc-python-reimplemented

# import part
import streamlit as st
from transformers import pipeline

# Optional: convert Simplified -> Traditional (HK)
try:
	from opencc import OpenCC
	_cc = OpenCC("s2hk")  # use "s2t" for generic Traditional
except Exception:
	_cc = None


# function part
@st.cache_resource
def get_sentiment_pipeline(model_name: str):
	return pipeline(
		"text-classification",
		model=model_name,
		tokenizer=model_name,
		return_all_scores=False,
	)


@st.cache_resource
def get_summarization_pipeline(model_name: str):
	return pipeline(
		"summarization",
		model=model_name,
		tokenizer=model_name,
		framework="pt",
	)


def normalize_text(title: str, text: str) -> str:
	title = (title or "").strip()
	text = (text or "").strip()
	return (title + " " + text).strip()


def map_sentiment_label(raw_label: str) -> str:
	label_map = {
		"LABEL_0": "Positive",
		"LABEL_1": "Negative",
	}
	return label_map.get(raw_label, raw_label)


def sentimentClassifier(text: str, model_name: str):
	clf = get_sentiment_pipeline(model_name)
	pred = clf(text, truncation=True, max_length=512)[0]
	return pred  # e.g., {"label": "LABEL_0", "score": 0.98}


def summarizer(text: str, model_name: str) -> str:
	sum_pipe = get_summarization_pipeline(model_name)
	out = sum_pipe(
		text,
		max_length=120,
		min_length=30,
		truncation=True,
	)[0]
	summary_text = out.get("summary_text", "")
	if _cc is not None:
		summary_text = _cc.convert(summary_text)
	return summary_text


def output_msg(input_text: str, sentiment_label: str, summary_text: str):
	st.subheader("Output")
	with st.expander("Input text (click to expand)", expanded=False):
		st.write(input_text)
	st.write("Sentiment:", sentiment_label)
	st.write("Summary (Pipeline 2):")
	st.write(summary_text)


def main():
	st.set_page_config(
		page_title="Chinese Finance News Briefing",
		layout="wide",
		initial_sidebar_state="collapsed",
	)

	st.header("Title: News Sentiment + Summarization (Hugging Face Pipelines)")
	st.caption("Note: First run may take time to download models on Streamlit Cloud.")

	st.subheader("Input")
	col1, col2 = st.columns(2)
	with col1:
		title = st.text_input("title", value="")
	with col2:
		text = st.text_input("text", value="")
	input_text = normalize_text(title, text)

	st.subheader("Models")
	sentiment_model = st.text_input(
		"Sentiment model (fine-tuned):",
		"mingflam/final_sentiment_model_v2",
	)
	summary_model = st.text_input(
		"Summarization model:",
		"heack/HeackMT5-ZhSum100k",
	)

	if st.button("Run", type="primary"):
		if input_text.strip() == "":
			st.warning("Please enter title/text.")
			return

		with st.spinner("Running sentiment..."):
			sentiment_pred = sentimentClassifier(input_text, sentiment_model)
			raw_label = sentiment_pred.get("label", "")
			final_label = map_sentiment_label(raw_label)

		# Show input + sentiment first
		output_msg(input_text, final_label, "")

		with st.spinner("Running summarization (this can take a while on Streamlit Cloud CPU)..."):
			summary_text = summarizer(input_text, summary_model)

		# Re-render output with summary
		output_msg(input_text, final_label, summary_text)


# main part
if __name__ == "__main__":
	main()
