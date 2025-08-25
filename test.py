import os
import asyncio
import json
from typing import Dict, Any
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
import subprocess
from typing import TypedDict, Annotated
import operator

from dotenv import load_dotenv

# Load .env file
load_dotenv()


# ----------------------------
# 1. AppState
# ----------------------------
class AppState(TypedDict):
    messages: Annotated[list[HumanMessage | AIMessage], operator.add]
    theme_files: Dict[str, Any]


# ----------------------------
# 2. Environment Setup
# ----------------------------
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

model = ChatOpenAI(
    model="google/gemini-2.5-pro",
    temperature=0.7,
    api_key=api_key,
    base_url=base_url,
)

client = MultiServerMCPClient(
    {
        "Figma Dev Mode MCP": {
            "url": "http://127.0.0.1:3845/mcp",
            "transport": "streamable_http",
        }, 
        "html.to.design": {
            "command": "uvx",
            "args": [
                "mcp-proxy",
                "--transport", "streamablehttp",
                "https://h2d-mcp.divriots.com/8d88243f-6717-4fbc-a399-3194e93a1955/mcp"
            ],
            "transport": "stdio"
        }
    }
)


# ----------------------------
# 3. Graph Nodes
# ----------------------------

# Discover MCP tools
async def setup_tools():
    tools = await client.get_tools()
    print("✅ Discovered MCP tools:", [t.name for t in tools])
    return tools


# Node: call_model
async def call_model(state: AppState) -> AppState:
        messages = state["messages"]
        response = await model_with_tools.ainvoke(messages)
        # print(f"AI model response in call_model: {response.content}")
        return {"messages": messages + [response]}



# Branch: should_continue
def should_continue(state: AppState):
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    elif hasattr(messages, "figma"):
        return "generate_theme"
    return END



# Node: generate_theme
async def generate_theme(state: AppState) -> AppState:
    """Generate Shopify theme files from Figma design."""
    messages = state["messages"]
    figma_url = next((msg.content for msg in messages if "figma.com" in msg.content), None)

    if not figma_url:
        error_msg = "❌ No Figma URL found in the conversation."
        print(error_msg)
        state["messages"].append(AIMessage(content=error_msg))
        return state

    prompt = (
        f"Using the Figma design at {figma_url}, analyze the design and generate Shopify theme files in Liquid format. "
        "Include at least templates/index.liquid, sections/hero.liquid, assets/style.css, and assets/script.js. "
        "Do not generate React, JSX, or any non-Shopify code. "
        "Return a JSON dictionary with file paths as keys and file contents as values. "
        "Ensure compatibility with Shopify's theme structure. "
        "Also include config/settings_schema.json."
    )

    messages.append(HumanMessage(content=prompt))
    response = await model_with_tools.ainvoke(messages)


    print(f"📝 AI model raw response: {response}")
    print(f"📝 AI model raw response: {response.content}")

    if response.content:
        try:
            raw_content = response.content.strip()

            # --- 🩹 FIX: remove markdown fences if present ---
            if raw_content.startswith("```"):
                import re
                raw_content = re.sub(r"^```[a-zA-Z0-9]*\n", "", raw_content)
                raw_content = re.sub(r"\n```$", "", raw_content)

            theme_files = json.loads(raw_content)

            # Add minimal defaults if missing
            if "config/settings_schema.json" not in theme_files:
                theme_files["config/settings_schema.json"] = json.dumps([
                    {
                        "name": "theme_info",
                        "theme_name": "Generated Theme",
                        "theme_version": "1.0.0",
                        "theme_author": "LangGraph"
                    }
                ])
            if "layout/theme.liquid" not in theme_files:
                theme_files["layout/theme.liquid"] = (
                    "<!DOCTYPE html>\n"
                    "<html>\n"
                    "<head>\n"
                    "  {{ content_for_header }}\n"
                    "  <link rel='stylesheet' href='{{ 'style.css' | asset_url }}'>\n"
                    "</head>\n"
                    "<body>\n"
                    "  {{ content_for_layout }}\n"
                    "  <script src='{{ 'script.js' | asset_url }}'></script>\n"
                    "</body>\n"
                    "</html>"
                )

            state["theme_files"] = theme_files
            state["messages"].append(AIMessage(content="✅ Shopify theme files generated successfully."))

            # Write theme files to local disk
            save_theme_files(theme_files)

        except json.JSONDecodeError as e:
            error_msg = f"❌ Failed to parse generated theme files: {e}"
            print(error_msg)
            state["messages"].append(AIMessage(content=error_msg))
    else:
        error_msg = "❌ No theme files generated from Figma design. AI response was empty."
        print(error_msg)
        state["messages"].append(AIMessage(content=error_msg))

    return state



# Node to push theme to shopify
def push_theme(state: AppState) -> AppState:
    store_name = "trestingpqr"
    theme_dir = os.path.join(os.getcwd(), "theme")
    if not os.path.exists(theme_dir):
        print("Theme directory does not exist, using default directory")
        theme_dir = "/Users/macbookair-unifynd/langgraph-workflow/shopify-theme"
    cmd = ["shopify", "theme", "push", "--store", store_name]

    try:
        result = subprocess.Popen(cmd, cwd=theme_dir)
        result.wait()
        print("✅ Shopify theme push succeeded")
        print(result.stdout)
        if result.stderr:
            print("⚠️ Shopify CLI warnings:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("❌ Shopify theme push failed")
        print(e.stdout)
        print(e.stderr)

    return state


# ----------------------------
# 4. Helper: Save theme files
# ----------------------------
def save_theme_files(theme_files: Dict[str, str], base_dir: str = "theme"):
    os.makedirs(base_dir, exist_ok=True)
    for path, content in theme_files.items():
        full_path = os.path.join(base_dir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"📂 Theme files saved to: {os.path.abspath(base_dir)}")


# ----------------------------
# 5. Main Entry
# ----------------------------
async def main():
    global model_with_tools
    tools = await setup_tools()

    # Bind tools to model
    model_with_tools = model.bind_tools(tools)

    tool_node = ToolNode(tools)

    # --- Build Graph ---
    builder = StateGraph(AppState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_node("generate_theme", generate_theme)
    builder.add_node("push_theme", push_theme)  

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue)
    builder.add_edge("tools", "call_model")
    builder.add_edge("generate_theme", "push_theme")
    builder.add_edge("push_theme", END)

    graph = builder.compile()

    # --- Interactive loop ---
    chat_history = []

    while True:
        user_query = input("Enter your Query: ")
        if user_query.lower() in ["quit", "exit", "q"]:
            print("👋 Goodbye!")
            break

        chat_history.append(HumanMessage(content=user_query))

        result = await graph.ainvoke({"messages": chat_history, "theme_files": {}})

        # Append assistant reply
        assistant_msg = result["messages"][-1]
        chat_history.append(assistant_msg)

        print("Assistant:", assistant_msg.content)


if __name__ == "__main__":
    asyncio.run(main())


