"""Custom ChatGPT - a ChatGPT-style web app (Streamlit + LangChain).

Run:  streamlit run app.py
The OpenAI API key is entered on the login screen and lives only in st.session_state.
"""
import html
import json
import os
import time
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
)

APP_NAME = "Custom ChatGPT"
CHATS_FILE = Path(__file__).parent / "chats.json"
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_SYSTEM = "You are a helpful assistant."

st.set_page_config(page_title=APP_NAME, page_icon="💬", layout="centered",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- CSS
CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600;6..72,700&display=swap");
/* palette: mocha brown #6F4E37 | platinum silver #E5E4E2 | pale lavender #E6E0F0 */
/* ---- background ---- */
.stApp{background:linear-gradient(160deg,#EEEDEB 0%,#E5E4E2 55%,#DAD8D5 100%);background-attachment:fixed;
  font-family:Newsreader,Georgia,serif;color:#3B2618}
header[data-testid="stHeader"]{background:transparent}
/* hide only Deploy + main menu; the sidebar toggle stays visible */
[data-testid="stAppDeployButton"],.stDeployButton,#MainMenu,[data-testid="stMainMenu"]{display:none !important}
[data-testid="stExpandSidebarButton"],[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"]{visibility:visible !important;display:flex !important;color:#3B2618 !important}
[data-testid="stExpandSidebarButton"] *,[data-testid="stSidebarCollapsedControl"] *{color:#3B2618 !important}

/* ---- centered reading column ---- */
.block-container{max-width:780px !important;padding-top:2.2rem;padding-bottom:7rem}
[data-testid="stBottom"],[data-testid="stBottom"]>div{background:transparent !important}
[data-testid="stBottomBlockContainer"]{max-width:780px !important;margin:0 auto !important;
  padding-left:1rem !important;padding-right:1rem !important}

/* ---- chat input ---- */
[data-testid="stChatInput"]>div{background:rgba(230,224,240,.75) !important;
  border:1.5px solid #6F4E37 !important;border-radius:28px !important;
  box-shadow:0 8px 28px rgba(111,78,55,.14)}
[data-testid="stChatInput"] textarea{color:#3B2618 !important;font-family:Newsreader,Georgia,serif !important;font-size:1.02rem}
[data-testid="stChatInput"] textarea::placeholder{color:rgba(59,38,24,.62)}
[data-testid="stChatInput"] button{background:#6F4E37 !important;color:#E6E0F0 !important;border-radius:50% !important}
.st-key-hero_input [data-testid="stChatInput"]>div{min-height:92px;border-radius:32px !important}
.st-key-hero_input [data-testid="stChatInput"] textarea{font-size:1.15rem}

/* ---- messages ---- */
[data-testid="stChatMessageAvatarUser"],[data-testid="stChatMessageAvatarAssistant"]{display:none !important}
[data-testid="stChatMessage"]{background:transparent;padding:.4rem 0;line-height:1.7;color:#3B2618}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  background:rgba(230,224,240,.95);border:1px solid rgba(111,78,55,.45);
  border-radius:20px;padding:.8rem 1.1rem;margin:.6rem 0}

/* ---- greeting + login ---- */
.greeting{text-align:center;font-size:2.8rem;line-height:1.2;margin:0 0 1.6rem;color:#5A3E2B;font-weight:700}
.login-title{text-align:center;font-size:2.3rem;margin:4vh 0 .2rem;color:#5A3E2B;font-weight:700}
.login-sub{text-align:center;color:#5A3E2B;font-size:1.05rem;margin-bottom:1.4rem}
[data-testid="stForm"]{background:rgba(230,224,240,.7);border:1.5px solid rgba(59,38,24,.62);
  border-radius:22px;padding:1.6rem;backdrop-filter:blur(10px)}

/* ---- inputs ---- */
[data-baseweb="input"],[data-baseweb="textarea"]{background:rgba(255,255,255,.7) !important;
  border:1px solid rgba(111,78,55,.3) !important;border-radius:14px !important;overflow:hidden}
[data-baseweb="base-input"]{background:transparent !important;border:none !important}
.stTextInput input,.stTextArea textarea{background:transparent !important;color:#3B2618 !important;
  -webkit-text-fill-color:#3B2618 !important;border:none !important;font-family:Newsreader,Georgia,serif !important}
.stTextInput input::placeholder,.stTextArea textarea::placeholder{color:rgba(78,54,38,.5) !important;-webkit-text-fill-color:rgba(78,54,38,.5) !important}
[data-testid="stTextInputRootElement"] button{background:transparent !important;color:#6F4E37 !important}
[data-testid="stTextInputRootElement"] svg{fill:#6F4E37 !important;color:#6F4E37 !important}
/* labels, captions, expander titles: always dark */
[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] *,label,label p{color:#3B2618 !important}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *{color:#5A3E2B !important;opacity:1 !important}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary *{color:#3B2618 !important}
[data-testid="stSpinner"] *{color:#3B2618 !important}
.stButton>button,[data-testid="stFormSubmitButton"]>button{font-family:Newsreader,Georgia,serif;border-radius:14px}
[data-testid="stFormSubmitButton"]>button{background:#6F4E37;color:#E6E0F0;border:none;font-weight:700}
[data-testid="stFormSubmitButton"]>button:hover{background:#5A3E2B;color:#E6E0F0}

/* ---- sidebar ---- */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#EFEBF6 0%,#E6E0F0 100%);
  border-right:1px solid rgba(111,78,55,.4)}
.brand{font-size:1.45rem;font-weight:700;color:#6F4E37;margin:.2rem 0 .9rem}
.recents-label{font-size:.95rem;font-weight:700;color:#5A3E2B;margin:.8rem 0 .2rem}
.st-key-new_chat button{background:#6F4E37;color:#E6E0F0;border:none;font-weight:700;width:100%}
.st-key-new_chat button:hover{background:#5A3E2B;color:#E6E0F0}
[data-testid="stSidebar"] .stButton>button{background:transparent;color:#3B2618;border:1px solid transparent;
  justify-content:flex-start;text-align:left;width:100%}
[data-testid="stSidebar"] .stButton>button p{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
[data-testid="stSidebar"] .stButton>button:hover{background:rgba(111,78,55,.07);border-color:rgba(111,78,55,.22)}
[class*="st-key-active_"] button{background:rgba(111,78,55,.14) !important;border-color:#6F4E37 !important}
[class*="st-key-del_"] button{justify-content:center !important;opacity:.55;padding:0;min-height:2.2rem}
[class*="st-key-del_"] button:hover{opacity:1;color:#b3261e}
.st-key-logout button{border:1px solid rgba(111,78,55,.35) !important;justify-content:center !important}
[data-testid="stExpander"]{border:1px solid rgba(111,78,55,.45);border-radius:14px;background:rgba(255,255,255,.35)}

/* ---- clarity layer: solid light inputs, strong contrast ---- */
:root{color-scheme:light}
.stApp .stTextInput div[data-baseweb="input"],.stApp .stTextInput div[data-baseweb="base-input"],
.stApp [data-testid="stTextInputRootElement"],.stApp [data-testid="stTextInputRootElement"]>div,
.stApp .stTextArea div[data-baseweb="textarea"],.stApp .stTextArea div[data-baseweb="base-input"]{
  background:#FBFAFD !important;background-color:#FBFAFD !important}
.stApp [data-baseweb="input"],.stApp [data-baseweb="textarea"]{border:1.5px solid #6F4E37 !important;
  border-radius:14px !important;box-shadow:none !important}
.stApp [data-baseweb="input"]:focus-within,.stApp [data-baseweb="textarea"]:focus-within{
  box-shadow:0 0 0 3px rgba(111,78,55,.28) !important}
.stApp .stTextInput input,.stApp .stTextArea textarea{background:transparent !important;color:#2E1D12 !important;
  -webkit-text-fill-color:#2E1D12 !important;font-size:1.05rem !important}
.stApp .stTextInput input::placeholder,.stApp .stTextArea textarea::placeholder{
  color:rgba(59,38,24,.62) !important;-webkit-text-fill-color:rgba(59,38,24,.62) !important}
.stApp [data-testid="stWidgetLabel"] p{font-weight:700 !important;font-size:1.02rem !important;color:#2E1D12 !important}
.stApp [data-testid="stFormSubmitButton"]>button{min-height:3rem;font-size:1.05rem}
.stApp .st-key-new_chat button{min-height:2.8rem;font-size:1.02rem}

/* ---- sidebar fixes: light expander + textarea, left-aligned chat rows ---- */
.stApp .stTextInput input,.stApp .stTextArea textarea{background:#FBFAFD !important}
.stApp [data-testid="stExpander"] details,.stApp [data-testid="stExpander"] summary{
  background:#F3EFF8 !important;color:#3B2618 !important;border-radius:14px}
.stApp [data-testid="stExpander"] summary:hover{background:#EAE4F3 !important}
.stApp [data-testid="stExpander"] summary svg{fill:#3B2618 !important;color:#3B2618 !important}
.stApp [data-testid="stExpander"] details[open] summary{border-bottom-left-radius:0;border-bottom-right-radius:0}
[class*="st-key-chat_"] button>div,[class*="st-key-active_"] button>div{width:100%;justify-content:flex-start !important}
[class*="st-key-chat_"] button [data-testid="stMarkdownContainer"],
[class*="st-key-active_"] button [data-testid="stMarkdownContainer"]{width:100%;text-align:left !important}
.stApp [class*="st-key-del_"] button{background:transparent !important;color:#5A3E2B !important;
  border:1px solid rgba(111,78,55,.3) !important;min-height:2.2rem}
.stApp [class*="st-key-del_"] button:hover{background:rgba(179,38,30,.1) !important;color:#b3261e !important;border-color:#b3261e !important}

/* ================= modern + motion layer ================= */
@keyframes rise{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}
@keyframes blurIn{from{opacity:0;filter:blur(14px);transform:translateY(10px) scale(.98)}to{opacity:1;filter:blur(0);transform:none}}
@keyframes drift1{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(7vw,5vh) scale(1.15)}}
@keyframes drift2{0%,100%{transform:translate(0,0) scale(1.05)}50%{transform:translate(-6vw,-6vh) scale(.95)}}
@keyframes dots{0%,80%,100%{transform:scale(.55);opacity:.35}40%{transform:scale(1);opacity:1}}
@keyframes shine{from{transform:translateX(-130%) skewX(-20deg)}to{transform:translateX(260%) skewX(-20deg)}}

/* ambient background: two slow soft colour blobs behind everything */
.stApp::before,.stApp::after{content:"";position:fixed;inset:-12%;z-index:0;pointer-events:none}
.stApp::before{background:radial-gradient(38vw 38vw at 14% 22%,rgba(214,198,242,.70),transparent 62%),
  radial-gradient(34vw 34vw at 86% 78%,rgba(111,78,55,.16),transparent 62%);animation:drift1 28s ease-in-out infinite}
.stApp::after{background:radial-gradient(30vw 30vw at 78% 14%,rgba(236,230,248,.95),transparent 62%),
  radial-gradient(28vw 28vw at 18% 86%,rgba(150,118,86,.15),transparent 62%);animation:drift2 34s ease-in-out infinite}
[data-testid="stMain"]{position:relative;z-index:1}

::selection{background:rgba(111,78,55,.35);color:#2E1D12}
*::-webkit-scrollbar{width:8px;height:8px}
*::-webkit-scrollbar-thumb{background:rgba(111,78,55,.35);border-radius:8px}

/* hero entrance: greeting words blur into focus, then input, then cards */
.rv{display:inline-block;animation:blurIn 1s cubic-bezier(.2,.7,.2,1) both}
.greeting{font-size:3.4rem;letter-spacing:-.01em;font-weight:600}
.login-title{font-weight:600;letter-spacing:-.01em}
.login-sub{animation:rise .8s .45s both}
[data-testid="stForm"]{animation:rise .9s .55s cubic-bezier(.2,.7,.2,1) both;box-shadow:0 20px 60px rgba(111,78,55,.15)}
.st-key-hero_input{animation:rise .8s .55s cubic-bezier(.2,.7,.2,1) both}
[class*="st-key-chip_"]{animation:rise .7s cubic-bezier(.2,.7,.2,1) both}
.st-key-chip_0{animation-delay:.95s}.st-key-chip_1{animation-delay:1.05s}
.st-key-chip_2{animation-delay:1.15s}.st-key-chip_3{animation-delay:1.25s}

/* suggestion cards */
[class*="st-key-chip_"] button{background:rgba(255,255,255,.58) !important;border:1px solid rgba(111,78,55,.3) !important;
  border-radius:18px !important;min-height:4.2rem;padding:.9rem 1.2rem !important;color:#3B2618 !important;
  backdrop-filter:blur(10px);transition:transform .22s ease,box-shadow .22s ease,border-color .22s ease,background .22s ease}
[class*="st-key-chip_"] button>div{width:100%;justify-content:flex-start !important}
[class*="st-key-chip_"] button [data-testid="stMarkdownContainer"]{text-align:left !important;width:100%}
[class*="st-key-chip_"] button p{font-size:1.02rem;line-height:1.35}
[class*="st-key-chip_"] button:hover{transform:translateY(-4px);background:rgba(255,255,255,.9) !important;
  border-color:#6F4E37 !important;box-shadow:0 16px 34px rgba(111,78,55,.22)}
[class*="st-key-chip_"] button:active{transform:translateY(-1px) scale(.99)}

/* glass chat input */
[data-testid="stChatInput"]>div{background:rgba(255,255,255,.62) !important;backdrop-filter:blur(16px);
  -webkit-backdrop-filter:blur(16px);transition:box-shadow .3s ease,transform .3s ease}
[data-testid="stChatInput"]>div:focus-within{box-shadow:0 0 0 4px rgba(111,78,55,.16),0 16px 44px rgba(111,78,55,.24) !important;
  transform:translateY(-1px)}
[data-testid="stChatInput"] button{transition:transform .2s ease,box-shadow .2s ease}
[data-testid="stChatInput"] button:hover{transform:scale(1.12);box-shadow:0 6px 16px rgba(111,78,55,.35)}
[data-testid="stBottom"]::before{content:"";position:absolute;left:0;right:0;top:-46px;height:46px;pointer-events:none;
  background:linear-gradient(to bottom,rgba(229,228,226,0),rgba(229,228,226,.92))}

/* messages */
[data-testid="stChatMessage"]{animation:rise .45s cubic-bezier(.2,.7,.2,1) both}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  width:fit-content;max-width:86%;margin-left:auto;border-radius:22px 22px 6px 22px;
  box-shadow:0 6px 18px rgba(111,78,55,.12)}
.thinking{display:flex;gap:7px;padding:.7rem 0}
.thinking span{width:9px;height:9px;border-radius:50%;background:#6F4E37;animation:dots 1.2s infinite ease-in-out}
.thinking span:nth-child(2){animation-delay:.15s}.thinking span:nth-child(3){animation-delay:.3s}

/* buttons: lift, press, shine */
.stApp .stButton>button,.stApp [data-testid="stFormSubmitButton"]>button{
  position:relative;overflow:hidden;transition:transform .18s ease,box-shadow .18s ease,background .18s ease,border-color .18s ease}
.st-key-new_chat button:hover,[data-testid="stFormSubmitButton"]>button:hover{
  transform:translateY(-2px);box-shadow:0 10px 24px rgba(111,78,55,.32)}
.st-key-new_chat button:active,[data-testid="stFormSubmitButton"]>button:active{transform:translateY(0) scale(.98)}
.st-key-new_chat button::after,[data-testid="stFormSubmitButton"]>button::after{content:"";position:absolute;top:0;bottom:0;left:0;
  width:40%;background:linear-gradient(90deg,transparent,rgba(255,255,255,.38),transparent);transform:translateX(-130%) skewX(-20deg)}
.st-key-new_chat button:hover::after,[data-testid="stFormSubmitButton"]>button:hover::after{animation:shine .8s ease}

/* sidebar rows */
[class*="st-key-chat_"] button:hover{transform:translateX(4px)}
[class*="st-key-active_"] button{box-shadow:inset 4px 0 0 #6F4E37}
[class*="st-key-chat_"] button,[class*="st-key-active_"] button{transition:transform .18s ease,background .18s ease,border-color .18s ease}
.st-key-logout button:hover{background:rgba(111,78,55,.1) !important}

@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none !important;transition:none !important}}

/* ================= fixes: readable chat text, solid New chat, clean bottom, visible sidebar toggle ================= */
/* chat text always dark (config.toml may still carry old light text colour) */
.stApp [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
.stApp [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] *{color:#3B2618 !important}
.stApp [data-testid="stChatMessage"] a{color:#6F4E37 !important;text-decoration:underline}
.stApp [data-testid="stChatMessage"] pre,.stApp [data-testid="stChatMessage"] pre *{background:#F3EFF8 !important;color:#2E1D12 !important}
.stApp [data-testid="stChatMessage"] :not(pre)>code{background:rgba(111,78,55,.12) !important;color:#2E1D12 !important;border-radius:6px;padding:.1rem .35rem}
.stApp [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p{font-size:1.06rem;line-height:1.75}

/* New chat: solid mocha (beats the generic sidebar button rule) */
.stApp .st-key-new_chat button{background:#6F4E37 !important;color:#E6E0F0 !important;border:none !important;
  justify-content:center !important;font-weight:700;min-height:2.8rem}
.stApp .st-key-new_chat button p,.stApp .st-key-new_chat button *{color:#E6E0F0 !important}
.stApp .st-key-new_chat button:hover{background:#5A3E2B !important}
[data-testid="stSidebar"] .st-key-new_chat button>div{justify-content:center !important}

/* bottom bar: no flat band, soft blur fade instead */
[data-testid="stBottom"],[data-testid="stBottom"]>div,[data-testid="stBottomBlockContainer"]{background:transparent !important}
[data-testid="stBottom"]::before{top:-44px;bottom:0;height:auto;z-index:-1;
  background:linear-gradient(to bottom,rgba(229,228,226,0),rgba(229,228,226,.5));
  backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  -webkit-mask-image:linear-gradient(to bottom,transparent,#000 55%);mask-image:linear-gradient(to bottom,transparent,#000 55%)}

/* sidebar: tighter top, always-visible collapse icon */
[data-testid="stSidebarUserContent"]{padding-top:.4rem !important}
[data-testid="stSidebarCollapseButton"],[data-testid="stSidebarCollapseButton"] *{opacity:1 !important;color:#3B2618 !important;fill:#3B2618 !important}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------- persistence
def load_chats() -> dict:
    try:
        data = json.loads(CHATS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    chats = {}
    if isinstance(data, dict):
        for cid, c in data.items():
            if isinstance(c, dict) and isinstance(c.get("messages"), list):
                chats[cid] = {
                    "title": c.get("title") or "New chat",
                    "system": c.get("system") or DEFAULT_SYSTEM,
                    "messages": [m for m in c["messages"] if isinstance(m, dict) and "role" in m and "content" in m],
                    "updated": float(c.get("updated") or 0),
                }
    return chats


def save_chats() -> None:
    """Write every non-empty chat to chats.json (atomic replace)."""
    data = {cid: c for cid, c in st.session_state.chats.items() if c["messages"]}
    tmp = CHATS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, CHATS_FILE)


# --------------------------------------------------------------------- helpers
def validate_key(key: str):
    """Return None if the key works, otherwise a human-readable error."""
    try:
        OpenAI(api_key=key, max_retries=0, timeout=20).models.list()
    except AuthenticationError:
        return "Invalid API key. Check it and try again."
    except APIConnectionError:
        return "Could not reach OpenAI. Check your internet connection and try again."
    except APIStatusError as e:
        return f"OpenAI rejected the request (status {e.status_code}). Check your key and account."
    except Exception as e:  # noqa: BLE001
        return f"Could not validate the key: {e}"
    return None


def new_chat() -> None:
    chats = st.session_state.chats
    cur = chats.get(st.session_state.get("active"))
    if cur is not None and not cur["messages"]:
        return  # already on an empty chat
    cid = uuid.uuid4().hex
    chats[cid] = {"title": "New chat", "system": DEFAULT_SYSTEM, "messages": [], "updated": time.time()}
    st.session_state.active = cid


def open_chat(cid: str) -> None:
    st.session_state.active = cid
    st.session_state.pending = False


def delete_chat(cid: str) -> None:
    st.session_state.chats.pop(cid, None)
    save_chats()
    if st.session_state.active == cid:
        st.session_state.pending = False
        st.session_state.active = None
        new_chat()


def logout() -> None:
    st.session_state.clear()


def token_stream(llm, lc_messages, on_first=None):
    first = True
    for chunk in llm.stream(lc_messages):
        text = chunk.content
        if isinstance(text, list):
            text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in text)
        if text:
            if first and on_first:
                on_first()  # remove the "thinking" dots once the first token arrives
            first = False
            yield text


def reveal(text: str, step: float = 0.09) -> str:
    """Wrap each word in a span so the words blur into focus one after another."""
    words = html.escape(text).split(" ")
    return " ".join(f'<span class="rv" style="animation-delay:{0.1 + i * step:.2f}s">{w}</span>'
                    for i, w in enumerate(words))


CHIPS = [
    ("✍️ Write a polite email", "Write a polite email asking my manager for a deadline extension."),
    ("💡 Explain something simply", "Explain how neural networks work in simple terms with an everyday analogy."),
    ("🧑‍💻 Plan my Python learning", "Give me a practical 3-day study plan to start learning Python."),
    ("🚀 Portfolio project ideas", "Suggest 5 creative portfolio project ideas for a beginner developer."),
]
THINKING = '<div class="thinking"><span></span><span></span><span></span></div>'


def submit_prompt(text: str) -> None:
    """Add a user message to the active chat and mark it for answering."""
    chat = st.session_state.chats[st.session_state.active]
    chat["messages"].append({"role": "user", "content": text})
    if sum(1 for m in chat["messages"] if m["role"] == "user") == 1:
        chat["title"] = " ".join(text.split())[:40]
    chat["updated"] = time.time()
    save_chats()
    st.session_state.pending = True


# ------------------------------------------------------------------ login page
def login_screen() -> None:
    _, col, _ = st.columns([1, 6, 1])
    with col:
        st.markdown(f'<div class="login-title">{reveal("💬 " + APP_NAME)}</div>'
                    '<div class="login-sub">Sign in with your OpenAI API key to start chatting.</div>',
                    unsafe_allow_html=True)
        with st.form("login"):
            name = st.text_input("Your name (optional)", placeholder="e.g. Alex")
            key = st.text_input("OpenAI API key", type="password", placeholder="sk-...")
            submitted = st.form_submit_button("Enter", use_container_width=True)
        st.caption("Your key is kept only in this browser session's memory. It is never saved to disk.")
        if submitted:
            key = key.strip()
            if not key:
                st.error("Please enter your OpenAI API key.")
            else:
                with st.spinner("Checking your API key..."):
                    err = validate_key(key)
                if err:
                    st.error(err)
                else:
                    st.session_state.authed = True
                    st.session_state.api_key = key
                    st.session_state.user_name = name.strip()
                    st.rerun()


if not st.session_state.get("authed"):
    login_screen()
    st.stop()

# ------------------------------------------------------------------ app state
if "chats" not in st.session_state:
    st.session_state.chats = load_chats()
    st.session_state.active = None
    st.session_state.pending = False
if st.session_state.get("active") not in st.session_state.chats:
    st.session_state.active = None
    new_chat()

chat = st.session_state.chats[st.session_state.active]

# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f'<div class="brand">💬 {APP_NAME}</div>', unsafe_allow_html=True)
    st.button("＋ New chat", key="new_chat", on_click=new_chat, use_container_width=True)
    query = st.text_input("Search chats", key="search", placeholder="🔍 Search chats",
                          label_visibility="collapsed").strip().lower()

    st.markdown('<div class="recents-label">Recents</div>', unsafe_allow_html=True)
    items = sorted(
        ((cid, c) for cid, c in st.session_state.chats.items()
         if c["messages"] and query in c["title"].lower()),
        key=lambda x: x[1]["updated"], reverse=True)
    with st.container(height=220, border=False):
        if not items:
            st.caption("No chats found." if query else "Your chats will appear here.")
        for cid, c in items:
            a, b = st.columns([6, 1], gap="small", vertical_alignment="center")
            prefix = "active" if cid == st.session_state.active else "chat"
            a.button(c["title"], key=f"{prefix}_{cid}", on_click=open_chat, args=(cid,), use_container_width=True)
            b.button("✕", key=f"del_{cid}", on_click=delete_chat, args=(cid,), help="Delete chat")

    with st.expander("Settings"):
        st.text_input("Model", value=DEFAULT_MODEL, key="model")
        new_sys = st.text_area("System role (this chat)", value=chat["system"],
                               key=f"system_{st.session_state.active}", height=110)
        if new_sys != chat["system"]:
            chat["system"] = new_sys
            save_chats()

    st.button("Logout", key="logout", on_click=logout, use_container_width=True)

# ------------------------------------------------------------------ main area
msgs = chat["messages"]

if err := st.session_state.pop("last_error", None):
    st.error(err)

if not msgs:
    name = st.session_state.get("user_name", "")
    greeting = f"Hello, {name}" if name else "How can I help you today?"
    st.markdown(f'<div style="height:13vh"></div><div class="greeting">{reveal(greeting)}</div>', unsafe_allow_html=True)
    with st.container(key="hero_input"):
        prompt = st.chat_input("Ask anything", key="chat_prompt")
    st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)
    chip_cols = st.columns(2)
    for i, (label, p) in enumerate(CHIPS):
        chip_cols[i % 2].button(label, key=f"chip_{i}", on_click=submit_prompt, args=(p,),
                                use_container_width=True, help=p)
else:
    for m in msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    prompt = st.chat_input("Ask anything", key="chat_prompt")

if prompt and prompt.strip():
    submit_prompt(prompt.strip())
    st.rerun()

if st.session_state.pending and msgs and msgs[-1]["role"] == "user":
    error = None
    reply = ""
    with st.chat_message("assistant"):
        slot = st.empty()
        slot.markdown(THINKING, unsafe_allow_html=True)
        try:
            llm = ChatOpenAI(model=(st.session_state.get("model") or DEFAULT_MODEL).strip(),
                             api_key=st.session_state.api_key)
            lc = [SystemMessage(content=chat["system"] or DEFAULT_SYSTEM)]
            for m in msgs:
                lc.append(HumanMessage(content=m["content"]) if m["role"] == "user"
                          else AIMessage(content=m["content"]))
            reply = st.write_stream(token_stream(llm, lc, slot.empty))
        except Exception as e:  # noqa: BLE001
            slot.empty()
            error = f"The request failed: {e}"
    st.session_state.pending = False
    if error:
        msgs.pop()  # remove the failed user message
        if not msgs:
            chat["title"] = "New chat"
        save_chats()
        st.session_state.last_error = error
        st.rerun()
    else:
        msgs.append({"role": "assistant", "content": reply})
        chat["updated"] = time.time()
        save_chats()