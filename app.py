import os
import pandas as pd
import streamlit as st

from google import genai

from src.database import (
    init_db,
    save_interaction,
    save_feedback,
    stats,
    feedback_rows,
    recent_rows,
    csv_export
)

from src.ai import demo
from src.sentiment import sentiment
from src.analytics import dashboard


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Nexora AI",
    page_icon="💡",
    layout="wide"
)

# =====================================================
# 2. DATABASE INITIALIZATION
# =====================================================

init_db()


# =====================================================
# 3. GEMINI API CONFIGURATION
# =====================================================

try:
    API_KEY = (
        st.secrets.get("GOOGLE_API_KEY", "")
        or st.secrets.get("GEMINI_API_KEY", "")
    )
except Exception:
    API_KEY = ""

API_KEY = (
    API_KEY
    or os.getenv("GOOGLE_API_KEY", "")
    or os.getenv("GEMINI_API_KEY", "")
).strip()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

client = None

if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception:
        client = None


# =====================================================
# 4. AI RESPONSE GENERATION
# =====================================================

def generate_response(prompt):

    if client is None:
        return demo(prompt), "Demo mode · no Gemini API key"

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        if response.text:
            return response.text, f"Live response · {MODEL}"

        return "The AI returned an empty response.", "Empty response"

    except Exception as e:
        return (
            f"Gemini API error: {str(e)}",
            "Gemini API error"
        )


# =====================================================
# 5. SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_id" not in st.session_state:
    st.session_state.last_id = None

if "feedback_saved_ids" not in st.session_state:
    st.session_state.feedback_saved_ids = set()


# =====================================================
# 6. CUSTOM STYLING
# =====================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
    max-width: 1400px;
}

.hero {
    background: linear-gradient(
        115deg,
        #173b75,
        #2563eb,
        #38bdf8
    );
    padding: 28px;
    border-radius: 14px;
    color: white;
    margin-bottom: 22px;
}

.hero h1 {
    color: white;
    font-size: 34px;
}

.hero p {
    color: #e0f2fe;
    font-size: 16px;
}

div[data-testid="stMetric"] {
    background: #f8fafc;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 7. SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown("# 💡 Nexora AI")

    st.caption("AI response quality & feedback intelligence")

    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "AI Assistant",
            "Feedback Explorer",
            "Response Evaluation"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.caption("Educational prototype")

    if client:
        st.success("Gemini AI connected")
    else:
        st.info("Demo mode active")

    st.caption(
        "Do not enter confidential information."
    )


# =====================================================
# 8. APPLICATION HEADER
# =====================================================

st.markdown("""
<div class="hero">
    <h1>💡 Nexora AI</h1>
    <p>
        Turn AI conversations into measurable insights.
        Analyze response quality, collect feedback,
        understand sentiment and improve user experience.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 9. OVERVIEW DASHBOARD
# =====================================================

if page == "Overview":

    st.title("Feedback Intelligence Overview")

    st.write(
        "Monitor AI interactions, feedback quality, "
        "user satisfaction and sentiment."
    )

    dashboard()

    s = stats()

    st.divider()

    st.subheader("Project Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "**AI Interactions**\n\n"
            "Total questions processed by InsightAI."
        )

    with col2:
        st.info(
            "**User Satisfaction**\n\n"
            "Average feedback rating from users."
        )

    with col3:
        st.info(
            "**Sentiment Analysis**\n\n"
            "Understand positive, neutral and negative feedback."
        )


# =====================================================
# 10. AI ASSISTANT
# =====================================================

elif page == "AI Assistant":

    st.title("AI Assistant")

    st.write(
        "Ask a question, review the response, "
        "and rate its quality."
    )

    st.warning(
        "Do not enter passwords, OTPs, API keys, "
        "financial details or sensitive personal information."
    )

    # Display conversation history

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("mode"):
                st.caption(message["mode"])

    # User input

    prompt = st.chat_input("Ask InsightAI a question...")

    if prompt:

        # Display user message

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # Generate AI response

        with st.spinner("Generating response..."):

            reply, mode = generate_response(prompt)

            interaction_id = save_interaction(
                prompt,
                reply,
                mode
            )

        # Save assistant message

        st.session_state.messages.append({
            "role": "assistant",
            "content": reply,
            "mode": mode
        })

        st.session_state.last_id = interaction_id

        st.rerun()

    # =================================================
    # RESPONSE RATING
    # =================================================

    last_id = st.session_state.last_id

    if last_id is not None:

        if last_id in st.session_state.feedback_saved_ids:

            st.success("Feedback already submitted for this response.")

        else:

            st.divider()

            st.subheader("Rate the latest response")

            with st.form("response_feedback"):

                rating = st.slider(
                    "How helpful was this answer?",
                    min_value=1,
                    max_value=5,
                    value=4
                )

                helpful = st.radio(
                    "Did it address your question?",
                    [
                        "Yes",
                        "Partially",
                        "No"
                    ],
                    horizontal=True
                )

                category = st.selectbox(
                    "Feedback category",
                    [
                        "Accuracy",
                        "Relevance",
                        "Clarity",
                        "Completeness",
                        "Other"
                    ]
                )

                comment = st.text_area(
                    "Additional feedback (optional)",
                    placeholder="Share your experience..."
                )

                submitted = st.form_submit_button(
                    "Submit Feedback",
                    use_container_width=True
                )

                if submitted:

                    sentiment_text = (
                        comment if comment.strip()
                        else f"{helpful} {category}"
                    )

                    label, score = sentiment(sentiment_text)

                    save_feedback(
                        last_id,
                        rating,
                        helpful,
                        category,
                        comment,
                        label,
                        score
                    )

                    st.session_state.feedback_saved_ids.add(last_id)

                    st.success(
                        "Thank you! Your feedback has been recorded."
                    )

                    st.rerun()


# =====================================================
# 11. FEEDBACK EXPLORER
# =====================================================

elif page == "Feedback Explorer":

    st.title("Feedback Explorer")

    st.write(
        "Explore user feedback, ratings and sentiment patterns."
    )

    rows = feedback_rows()

    if not rows:

        st.info(
            "No feedback available yet. "
            "Use the AI Assistant to generate responses "
            "and submit feedback."
        )

    else:

        df = pd.DataFrame(rows)

        # Summary metrics

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Total Feedback", len(df))

        with c2:
            st.metric(
                "Average Rating",
                f"{df['rating'].mean():.2f}/5"
            )

        with c3:
            st.metric(
                "Positive Feedback",
                int((df["sentiment"] == "Positive").sum())
            )

        st.divider()

        # Filters

        st.subheader("Filter Feedback")

        f1, f2 = st.columns(2)

        with f1:

            selected_sentiment = st.selectbox(
                "Sentiment",
                ["All", "Positive", "Neutral", "Negative"]
            )

        with f2:

            selected_category = st.selectbox(
                "Category",
                ["All"] + sorted(
                    df["category"].dropna().unique().tolist()
                )
            )

        filtered = df.copy()

        if selected_sentiment != "All":

            filtered = filtered[
                filtered["sentiment"] == selected_sentiment
            ]

        if selected_category != "All":

            filtered = filtered[
                filtered["category"] == selected_category
            ]

        st.subheader("Feedback Records")

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

        # CSV download

        st.download_button(
            label="Download Feedback CSV",
            data=csv_export(),
            file_name="insightai_feedback.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.divider()

        # Sentiment distribution

        st.subheader("Sentiment Distribution")

        sentiment_counts = (
            filtered["sentiment"]
            .value_counts()
            .reindex(
                ["Positive", "Neutral", "Negative"],
                fill_value=0
            )
        )

        st.bar_chart(sentiment_counts)


# =====================================================
# 12. RESPONSE EVALUATION
# =====================================================

elif page == "Response Evaluation":

    st.title("Response Evaluation")

    st.write(
        "Evaluate AI response quality using user ratings, "
        "helpfulness and feedback."
    )

    rows = recent_rows()

    if not rows:

        st.info(
            "No AI interactions available yet. "
            "Start a conversation in AI Assistant."
        )

    else:

        df = pd.DataFrame(rows)

        # Evaluation metrics

        rated = df.dropna(subset=["rating"])

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Total Interactions",
                len(df)
            )

        with c2:
            st.metric(
                "Evaluated Responses",
                len(rated)
            )

        with c3:

            average = (
                rated["rating"].mean()
                if not rated.empty else 0
            )

            st.metric(
                "Average Quality Rating",
                f"{average:.2f}/5" if not rated.empty else "—"
            )

        st.divider()

        # Rating distribution

        st.subheader("Rating Distribution")

        if not rated.empty:

            rating_counts = (
                rated["rating"]
                .value_counts()
                .reindex([1, 2, 3, 4, 5], fill_value=0)
            )

            st.bar_chart(rating_counts)

        else:

            st.info("Ratings will appear after users submit feedback.")

        st.divider()

        # Individual evaluations

        st.subheader("Individual Response Evaluations")

        for row in rows:

            with st.expander(
                f"Interaction #{row['id']} · {row['created_at']}"
            ):

                st.markdown("**User Question**")

                st.write(row["question"])

                st.markdown("**AI Response**")

                st.write(row["answer"])

                st.divider()

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "User Rating",
                        f"{row['rating']}/5"
                        if row["rating"] is not None
                        else "Pending"
                    )

                    st.write(
                        "Helpful:",
                        row["helpful"] or "Not evaluated"
                    )

                with col2:

                    st.write(
                        "Sentiment:",
                        row["sentiment"] or "Pending"
                    )

                    st.write(
                        "Category:",
                        row["category"] or "Not specified"
                    )

                if row["comment"]:

                    st.markdown("**User Comment**")

                    st.write(row["comment"])


# =====================================================
# 13. FOOTER
# =====================================================

st.divider()

st.caption(
    "InsightAI | AI Response Quality & Feedback Intelligence "
    "| Educational Prototype"
)
