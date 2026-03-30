from enum import Enum
from pathlib import Path
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.agent.hf_llm import HF_LLM

LLM_MODEL_PATH = str(Path(__file__).parent / "storage" / "llm_models" / "gemma-3-4b-it")

# ---------------------------------------------------------------------------
# Conversation states
# ---------------------------------------------------------------------------

class State(str, Enum):
    INTRODUCE    = "introduce"
    GET_PROPERTY = "get_property"
    GET_SCHEDULE = "get_schedule"
    END_YES      = "end_yes"
    END_NO       = "end_no"

# Keywords used to detect yes / no intent from the customer
_YES_KEYWORDS = {
    "haa", "haan", "haanji", "hanji", "han", "hnji",
    "yes", "yep", "yeah",
    "ji", "jee", "bilkul", "zaroor",
    "chahiye", "karana", "karwana", "karwao", "karwa",
    "h", "ha",
}
_NO_KEYWORDS = {"nahi", "nhi", "no", "nai", "mat", "nahin", "nhin", "nah"}

def detect_intent(text: str) -> str:
    words = set(text.lower().split())
    if words & _NO_KEYWORDS:
        return "no"
    if words & _YES_KEYWORDS:
        return "yes"
    return "other"

def next_state(current: State, user_input: str) -> State:
    """Determine the next conversation state from the current state and user input."""
    if current == State.INTRODUCE:
        intent = detect_intent(user_input)
        if intent == "yes":
            return State.GET_PROPERTY
        if intent == "no":
            return State.END_NO
        return State.INTRODUCE  # unclear answer — stay and ask again
    if current == State.GET_PROPERTY:
        return State.GET_SCHEDULE
    if current == State.GET_SCHEDULE:
        return State.END_YES
    return current  # END_YES / END_NO are terminal

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = """
You are Rahul, calling from The Ritika Edit, an interior design company. You are on a phone call with a customer.

Rules:
- Speak only in Hindi. Use English only for words like "interior work", "property", "designer", "2BHK", "3BHK", "office", "date", "time".
- Output ONLY the words to be spoken. No labels, no translations, no extra text.
- Keep each reply to one or two sentences.
- Do not discuss pricing, materials, or design styles. If asked, say the designer will explain on the call.
"""

_STEP_INSTRUCTIONS = {
    State.INTRODUCE:    "Introduce yourself: your name is Rahul and you are calling from The Ritika Edit. Then ask if they have any requirement for interior work.",
    State.GET_PROPERTY: "The customer is interested in interior work. Ask about their property type (2BHK, 3BHK, villa, or office) and which city or area it is located in.",
    State.GET_SCHEDULE: "You have the property details. Tell the customer that a designer from The Ritika Edit will connect with them over a call. Ask for their preferred date and time for that call.",
    State.END_YES:      "The customer has given their preferred date and time. Thank them warmly and say goodbye.",
    State.END_NO:       "The customer has no interior work requirement. Thank them politely and say goodbye.",
}

_OPENING_LINE = "Hi, क्या मैं Mayank Anand से बात कर रहा हूँ?"

def build_system_prompt(state: State) -> str:
    return _BASE_SYSTEM_PROMPT + f"\nYour task for this response ONLY: {_STEP_INSTRUCTIONS[state]}"

# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Voicebot Evaluator", page_icon="🤖")
st.title("Voicebot Evaluator")

if "llm" not in st.session_state:
    with st.spinner("Loading model..."):
        st.session_state.llm = HF_LLM(model_path=LLM_MODEL_PATH)

if "state" not in st.session_state:
    st.session_state.state = State.INTRODUCE

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=build_system_prompt(st.session_state.state)),
        HumanMessage(content="[START]"),
        AIMessage(content=_OPENING_LINE),
    ]

# Render conversation history (skip system message and dummy [START])
for msg in st.session_state.messages[2:]:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.write(msg.content)

# Only accept input if conversation is not over
is_ended = st.session_state.state in (State.END_YES, State.END_NO)

if not is_ended:
    if user_input := st.chat_input("Reply to the bot..."):
        # Advance state based on user input
        new_state = next_state(st.session_state.state, user_input)
        st.session_state.state = new_state

        # Update system prompt to reflect the new state
        st.session_state.messages[0] = SystemMessage(content=build_system_prompt(new_state))

        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("..."):
                response = st.session_state.llm.chat(messages=st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=response.content))
            st.write(response.content)
else:
    st.info("Conversation ended.")

# Sidebar
with st.sidebar:
    st.header("Controls")
    st.markdown(f"**Current state:** `{st.session_state.state.value}`")

    if st.button("Reset conversation"):
        st.session_state.state = State.INTRODUCE
        st.session_state.messages = [
            SystemMessage(content=build_system_prompt(State.INTRODUCE)),
            HumanMessage(content="[START]"),
            AIMessage(content=_OPENING_LINE),
        ]
        st.rerun()

    with st.expander("Raw message log"):
        for msg in st.session_state.messages:
            st.markdown(f"**{type(msg).__name__}:** {msg.content}")
