import streamlit as st
import os
import sys
import io
import re
import pandas as pd
import uuid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Annotated, TypedDict, Literal

# LangGraph & LangChain
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# Supabase
from supabase import create_client, Client

# --- CONFIGURATION ---  # Required for server-side plotting
st.set_page_config(page_title="Autonomous Data Scientist", layout="wide")
st.title("🤖 Autonomous Data Scientist Agent")

# --- DATABASE SETUP ---
@st.cache_resource
def init_supabase():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    return create_client(url, key) if url and key else None

supabase = init_supabase()

def get_all_sessions():
    """Fetch unique session IDs for the sidebar."""
    if not supabase: return []
    try:
        res = supabase.table("chat_history").select("session_id").execute()
        unique_ids = list(set([row['session_id'] for row in res.data]))
        return sorted(unique_ids, reverse=True) 
    except: return []

def save_message(session_id, role, content):
    """Persist a single message to Supabase."""
    if supabase:
        supabase.table("chat_history").insert({
            "session_id": session_id, 
            "role": role, 
            "content": content
        }).execute()

def get_messages_from_db(session_id):
    """Convert Supabase rows into LangChain Message objects for rehydration."""
    if not supabase: return []
    res = supabase.table("chat_history").select("role, content").eq("session_id", session_id).order("created_at", desc=False).execute()
    
    messages = []
    for row in res.data:
        if row['role'] == 'user':
            messages.append(HumanMessage(content=row['content']))
        elif row['role'] == 'assistant':
            messages.append(AIMessage(content=row['content']))
    return messages

# --- SIDEBAR & SESSION MANAGEMENT ---
with st.sidebar:
    st.header("🗄️ Chat History")
    
    # New Chat Button
    if st.button("➕ New Chat", type="primary"):
        st.session_state["session_id"] = str(uuid.uuid4())[:8]
        st.session_state["messages"] = []
        st.rerun()

    existing_sessions = get_all_sessions()
    
    # Ensure session_id exists
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = existing_sessions[0] if existing_sessions else str(uuid.uuid4())[:8]
    
    # Handle Ghost IDs (ensure current ID is in list)
    current_id = st.session_state["session_id"]
    if current_id not in existing_sessions: 
        existing_sessions.insert(0, current_id)
    
    # Session Selector
    selected = st.selectbox(
        "Load Conversation:", 
        options=existing_sessions, 
        index=existing_sessions.index(current_id)
    )
    
    if selected != st.session_state["session_id"]:
        st.session_state["session_id"] = selected
        st.rerun()
    
    st.caption(f"ID: {st.session_state['session_id']}")
    st.divider()
    
    # Credentials & File Upload
    api_key = st.text_input("Gemini API Key", type="password")
    if api_key: os.environ["GOOGLE_API_KEY"] = api_key
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        path = f"/tmp/{st.session_state['session_id']}_{uploaded_file.name}"
        with open(path, "wb") as f: f.write(uploaded_file.getbuffer())
        st.session_state["dataset_path"] = path
        st.success(f"File Uploaded: {uploaded_file.name}")

# --- AGENT LOGIC ---
def local_executor(code):
    """Executes Python code in a restricted/sanitized environment."""
    old_out = sys.stdout
    new_out = io.StringIO()
    sys.stdout = new_out

    # ---1. SECURITY FIX START ---
    if "agent_globals" not in st.session_state:
        # 1. Create a whitelist of safe libraries
        safe_globals = {
            "pd": pd,
            "plt": plt,
            "io": io,
            "np": __import__("numpy"),
            "re": re,
            "math": __import__("math"),
            "__builtins__": __builtins__ # Required for basic python to work
        }
        
        # 2. Add specific session variables they explicitly NEED
        if "dataset_path" in st.session_state:
            safe_globals["dataset_path"] = st.session_state["dataset_path"]

        st.session_state["agent_globals"] = safe_globals
        
    # --- 2. SECURITY FIX: STRING GUARDRAILS ---
    # Even with restricted globals, 'exec' allows imports. We must ban them explicitly.
    forbidden_terms = ["import os", "import sys", "from os", "from sys", "subprocess", "shutil"]
    if any(term in code for term in forbidden_terms):
        sys.stdout = old_out # Reset stdout before returning
        return "Security Error: Direct system imports (os, sys, subprocess) are not allowed.", None
        
    try:
        # Run code in the restricted 'safe_globals' instead of full 'globals()'
        exec(code, st.session_state["agent_globals"])
        
        # (Rest of your plot capturing logic remains the same)
        if plt.get_fignums():
            plot_buf = io.BytesIO()
            plt.savefig(plot_buf, format='png')
            plot_buf.seek(0)
            plt.close()
            return new_out.getvalue(), plot_buf
            
        return new_out.getvalue(), None

    except Exception as e:
        return f"Error: {e}", None
    finally:
        sys.stdout = old_out

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def node(state: State):
    if "GOOGLE_API_KEY" not in os.environ: 
        return {"messages": [AIMessage("Please provide a Gemini API Key in the sidebar.")]}
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    sys_msg = SystemMessage(content=f"""
    You are an Autonomous Data Scientist.
    1. Output code strictly in ```python ... ``` blocks.
    2. Always ask for permission before running model training (e.g., .fit(), joblib.dump).
    3. Save models to '/tmp/model_{st.session_state['session_id']}.joblib'.
    4. Dataset path is: {st.session_state.get('dataset_path', 'None')}.
    5. If the user asks to plot, use matplotlib.pyplot.
    """)
    
    # Combine System Msg + History
    response = llm.invoke([sys_msg] + state["messages"])
    
    # Code Execution Check
    if "```python" in response.content:
        code_match = re.search(r"```python\n(.*?)```", response.content, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            
            # Human-in-the-loop for Training
            if any(x in code for x in [".fit", "joblib.dump", "RandomForest", "sklearn"]):
                decision = interrupt({"question": "Approve training code?", "code": code})
                if str(decision).lower() not in ["yes", "approve"]: 
                    return {"messages": [response, HumanMessage("Action Blocked by User.")]}
            
            # Execute Code
            text_out, plot_out = local_executor(code)
            
            # Handle Plot (Store in session state to render later)
            if plot_out:
                st.session_state["last_plot"] = plot_out
                return {"messages": [response, HumanMessage(f"Code Executed. Output: {text_out}\n[System: A plot was generated]")]}
            
            return {"messages": [response, HumanMessage(f"Code Executed. Output: {text_out}")]}
    
    # Permission Check (Text based)
    if "permission" in response.content.lower():
        decision = interrupt({"question": response.content})
        return {"messages": [response, HumanMessage(f"User Decision: {decision}")]}
    
    return {"messages": [response]}

# --- GRAPH BUILD ---
if "graph" not in st.session_state:
    workflow = StateGraph(State)
    workflow.add_node("agent", node)
    workflow.set_entry_point("agent")
    
    # Loop back to agent if the last message was a tool output (HumanMessage acting as tool output)
    # End if the last message was an Assistant message (waiting for user input)
    workflow.add_conditional_edges(
        "agent", 
        lambda s: "agent" if isinstance(s["messages"][-1], HumanMessage) else "__end__"
    )
    
    st.session_state["graph"] = workflow.compile(checkpointer=MemorySaver())

app = st.session_state["graph"]
config = {"configurable": {"thread_id": st.session_state["session_id"]}}

# --- REHYDRATION LOGIC (The Fix) ---
# Check if LangGraph memory is empty but DB has data
current_graph_state = app.get_state(config)
if not current_graph_state.values:
    db_msgs = get_messages_from_db(st.session_state["session_id"])
    if db_msgs:
        # Inject DB history into LangGraph RAM
        app.update_state(config, {"messages": db_msgs})

# --- UI RENDER ---
# Load display history from DB
hist_rows = supabase.table("chat_history").select("*").eq("session_id", st.session_state["session_id"]).order("created_at", desc=False).execute().data if supabase else []

for m in hist_rows:
    with st.chat_message("user" if m["role"]=="user" else "assistant"):
        st.write(m["content"])
        # If the message indicates a plot was made, we can't fetch the old plot from DB 
        # (unless we stored it). For now, we only show live plots.

# Show live plot if just generated
if "last_plot" in st.session_state and st.session_state["last_plot"]:
    st.image(st.session_state["last_plot"], caption="Generated Plot")
    # Clear it so it doesn't persist forever on screen
    st.session_state["last_plot"] = None

# --- CHAT INPUT ---
if prompt := st.chat_input():
    save_message(st.session_state["session_id"], "user", prompt)
    st.chat_message("user").write(prompt)
    
    with st.spinner("Agent is thinking..."):
        try:
            # Run the graph
            for event in app.stream({"messages": [HumanMessage(prompt)]}, config):
                if "agent" in event:
                    msg = event["agent"]["messages"][-1]
                    if isinstance(msg, AIMessage):
                        save_message(st.session_state["session_id"], "assistant", msg.content)
                        st.chat_message("assistant").write(msg.content)
                        
                        # If a plot was generated in this step, show it immediately
                        if "last_plot" in st.session_state and st.session_state["last_plot"]:
                             st.image(st.session_state["last_plot"])
                             st.session_state["last_plot"] = None
        except Exception as e:
            st.error(f"Execution Error: {str(e)}")

# --- INTERRUPTS (APPROVALS) ---
snap = app.get_state(config)
if snap.next and snap.tasks[0].interrupts:
    intr = snap.tasks[0].interrupts[0].value
    st.warning("⚠️ APPROVAL NEEDED")
    st.write(f"**Question:** {intr.get('question')}")
    if intr.get('code'):
        st.code(intr.get('code'), language='python')
        
    col1, col2 = st.columns(2)
    if col1.button("✅ Approve"):
        with st.spinner("Resuming..."):
            for ev in app.stream(Command(resume="yes"), config): 
                if "agent" in ev and isinstance(ev["agent"]["messages"][-1], AIMessage):
                    m = ev["agent"]["messages"][-1]
                    save_message(st.session_state["session_id"], "assistant", m.content)
                    st.chat_message("assistant").write(m.content)
        st.rerun()
        
    if col2.button("❌ Reject"):
        with st.spinner("Rejecting..."):
            list(app.stream(Command(resume="no"), config))
        st.rerun()

# --- DOWNLOAD MODEL ---
model_path = f"/tmp/model_{st.session_state['session_id']}.joblib"
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        st.download_button("⬇️ Download Trained Model", f, f"model_{st.session_state['session_id']}.joblib")
