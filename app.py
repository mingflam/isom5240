# import part
from transformers import pipeline
import streamlit as st


# function part
def sentimentClassifier(text, modelName):
	# Load sentiment pipeline (Pipeline 1)
	clf = pipeline(
		"text-classification",
		model=modelName,
		tokenizer=modelName,
		return_all_scores=False,
	)

	pred = clf(text, truncation=True, max_length=512)[0]
	return pred  # e.g., {"label": "LABEL_0", "score": 0.98}


def summarizer(text, modelName):
	# Load summarization pipeline (Pipeline 2)
	sum_pipe = pipeline(
		"summarization",
		model=modelName,
		tokenizer=modelName,
		framework="pt",
	)

	out = sum_pipe(
		text,
		max_length=80,
		min_length=20,
		truncation=True,
	)[0]
	return out  # e.g., {"summary_text": "..."}


def map_sentiment_label(raw_label):
	# Adjust this mapping to match your fine-tuned label meaning
	label_map = {
		"LABEL_0": "Positive",
		"LABEL_1": "Negative",
	}
	return label_map.get(raw_label, raw_label)


def output_msg(sentiment_pred, summary_text):
	# Display results
	raw_label = sentiment_pred.get("label", "")
	score = float(sentiment_pred.get("score", 0.0))
	final_label = map_sentiment_label(raw_label)

	st.write("Sentiment Result:")
	st.write(f"Prediction: {final_label}")
	st.write(f"Confidence: {score:.4f}")

	st.write("Summary Result:")
	st.write(summary_text)


def main():
	# Streamlit UI
	st.header("Title: News Sentiment + Summarization (Hugging Face Pipelines)")

	st.subheader("Input")
	user_text = st.text_area("Paste a news headline / short article:", height=150)

	st.subheader("Models")
	# Replace with your actual fine-tuned HF model id
	sentiment_model = st.text_input(
		"Sentiment model (fine-tuned):",
		"mingflam/final_sentiment_model_v2",
	)
	summary_model = st.text_input(
		"Summarization model:",
		"chiakya/T5-large-chinese-Summarization",
	)

	if st.button("Run"):
		if user_text.strip() == "":
			st.warning("Please input some text.")
			return

		# Pipeline 1
		sentiment_pred = sentimentClassifier(user_text, sentiment_model)

		# Build a simple “digest” prompt for pipeline 2
		raw_label = sentiment_pred.get("label", "")
		score = float(sentiment_pred.get("score", 0.0))
		final_label = map_sentiment_label(raw_label)

		digest = (
			f"新聞內容: {user_text}\n"
			f"情緒分類: {final_label} (confidence={score:.4f})\n"
			f"請總結重點:"
		)

		# Pipeline 2
		sum_out = summarizer(digest, summary_model)
		summary_text = sum_out.get("summary_text", "")

		# Output
		output_msg(sentiment_pred, summary_text)


# main part
if __name__ == "__main__":
	main()
