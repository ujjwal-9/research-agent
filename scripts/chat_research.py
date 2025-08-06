#!/usr/bin/env python3
"""
Interactive chat-based research workflow.

This script provides a conversational interface where you can chat with the research agent
to develop and refine your research plan before execution.
"""

import asyncio
import sys
import os
from pathlib import Path
import logging
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.chat_planner import ChatPlannerAgent, ChatSession
from research_workflow import ResearchWorkflowManager


class ChatResearchInterface:
    """Interactive chat interface for research planning."""

    def __init__(self):
        self.chat_agent = ChatPlannerAgent()
        self.chat_session: Optional[ChatSession] = None
        self.research_manager = None

        # Setup logging
        logging.basicConfig(level=logging.WARNING)  # Reduce noise during chat

    def print_welcome(self):
        """Print welcome message."""
        print("\n" + "=" * 70)
        print("🤖 Welcome to Interactive Research Chat!")
        print("=" * 70)
        print("I'm your research assistant. Let's have a conversation about what")
        print("you'd like to research, and I'll help you develop the perfect")
        print("research plan before we start the actual research.")
        print("\nType 'quit' or 'exit' to end the conversation.")
        print("Type 'plan' to see the current research plan (if generated).")
        print("Type 'start' to begin research once you're happy with the plan.")
        print("-" * 70)

    def print_separator(self, char="-", length=50):
        """Print a separator line."""
        print(char * length)

    async def start_conversation(self, initial_question: str):
        """Start the chat conversation."""

        self.print_welcome()

        try:
            # Start chat session
            print("🤖 Starting conversation...")
            self.chat_session = await self.chat_agent.start_chat_session(
                initial_question
            )

            # Get the initial response
            if self.chat_session.messages:
                # Find the last assistant message
                assistant_messages = [
                    msg for msg in self.chat_session.messages if msg.role == "assistant"
                ]
                if assistant_messages:
                    print(f"\n🤖 Assistant: {assistant_messages[-1].content}")

            # Start the conversation loop
            await self.conversation_loop()

        except Exception as e:
            print(f"\n❌ Error starting conversation: {e}")
            print("Please check your OpenAI API key is set in environment variables.")

    async def conversation_loop(self):
        """Main conversation loop."""

        while True:
            try:
                # Get user input
                print(f"\n👤 You: ", end="")
                user_input = input().strip()

                # Handle special commands
                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("\n👋 Thanks for chatting! Goodbye!")
                    break

                elif user_input.lower() == "plan":
                    self.show_current_plan()
                    continue

                elif user_input.lower() in ["start", "begin", "execute"]:
                    if await self.start_research():
                        break
                    continue

                elif user_input.lower() == "help":
                    self.show_help()
                    continue

                elif not user_input:
                    print("Please enter a message, or type 'help' for commands.")
                    continue

                # Continue the chat
                print("\n🤖 Assistant: ", end="")
                response = await self.chat_agent.continue_chat(user_input)
                print(response)

                # Check if a plan was generated
                if (
                    self.chat_agent.get_research_plan()
                    and not self.chat_session.plan_approved
                ):
                    print(f"\n💡 It looks like we've developed a research plan!")
                    print("Type 'plan' to review it, or 'start' to begin research.")

            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Let's continue our conversation...")

    def show_current_plan(self):
        """Show the current research plan."""
        plan = self.chat_agent.get_research_plan()

        if not plan:
            print("\n📋 No research plan has been generated yet.")
            print("Continue our conversation and I'll create one when you're ready!")
            return

        print("\n" + "=" * 60)
        print("📋 RESEARCH PLAN")
        print("=" * 60)
        print(f"**Question:** {plan.research_question}")
        print(f"\n**Methodology:**")
        print(plan.research_methodology)
        print(f"\n**Internal Search Queries ({len(plan.internal_search_queries)}):**")
        for i, query in enumerate(plan.internal_search_queries, 1):
            print(f"  {i}. {query}")
        print(f"\n**External Research Topics ({len(plan.external_search_topics)}):**")
        for i, topic in enumerate(plan.external_search_topics, 1):
            print(f"  {i}. {topic}")
        print(f"\n**Estimated Timeline:** {plan.estimated_timeline}")
        print(f"\n**Success Criteria:**")
        for criterion in plan.success_criteria:
            print(f"  • {criterion}")
        print("=" * 60)
        print(
            "Type 'start' to begin research, or continue chatting to refine the plan."
        )

    def show_help(self):
        """Show help information."""
        print("\n" + "=" * 50)
        print("🆘 HELP - Available Commands")
        print("=" * 50)
        print("• Just chat naturally to develop your research plan")
        print("• 'plan' - View the current research plan")
        print("• 'start' - Begin research with the current plan")
        print("• 'help' - Show this help message")
        print("• 'quit'/'exit' - End the conversation")
        print("\n💡 Tips:")
        print("• Be specific about what you want to research")
        print("• Tell me about the depth and scope you prefer")
        print("• Mention any specific aspects you're interested in")
        print("• Let me know the purpose of your research")
        print("=" * 50)

    async def start_research(self) -> bool:
        """Start the research workflow."""
        plan = self.chat_agent.get_research_plan()

        if not plan:
            print("\n⚠️ No research plan available yet.")
            print("Let's continue our conversation to develop one!")
            return False

        print(f"\n🚀 Starting research workflow...")
        print("This may take several minutes depending on the scope...")

        # Ask about code interpreter preference
        print("\n💡 Code Interpreter Options:")
        print(
            "The research can use Excel Code Interpreter for data analysis if relevant Excel files are found."
        )
        print(
            "👤 Would you like to use the Code Interpreter? (y/n, default=y): ", end=""
        )

        try:
            use_code_choice = input().strip().lower()
            use_code_interpreter = use_code_choice not in ["n", "no", "false"]

            if use_code_interpreter:
                print(
                    "✅ Code Interpreter enabled - Excel files will be analyzed if found"
                )
            else:
                print("🚫 Code Interpreter disabled - Excel analysis will be skipped")
        except (EOFError, KeyboardInterrupt):
            print("\nUsing default: Code Interpreter enabled")
            use_code_interpreter = True

        try:
            # Initialize research manager
            self.research_manager = ResearchWorkflowManager()

            # Execute research
            results = await self.research_manager.conduct_research(
                research_question=plan.research_question,
                user_requirements=self.chat_session.user_requirements or {},
                export_format="markdown",
                research_plan=plan,  # Pass the chat-generated plan
                use_code_interpreter=use_code_interpreter,
            )

            # Display results
            print(f"\n✅ Research completed successfully!")
            self.print_separator("=", 60)
            print("📊 RESEARCH RESULTS")
            self.print_separator("=", 60)

            summary = results.get("execution_summary", {})
            print(
                f"⏱️  Execution Time: {summary.get('total_execution_time', 0):.1f} seconds"
            )
            print(f"🤖 Agents Used: {summary.get('agents_executed', 0)}")
            print(f"📚 Sources Found: {summary.get('total_sources', 0)}")
            print(f"📄 Report Sections: {summary.get('report_sections', 0)}")

            # Show export info
            if "exported_reports" in results:
                print(f"\n📁 Generated Reports:")
                for format_type, content in results["exported_reports"].items():
                    if format_type.endswith("_file"):
                        file_path = content
                        if os.path.exists(file_path):
                            file_size = os.path.getsize(file_path)
                            print(
                                f"📄 {format_type.replace('_file', '').upper()}: {file_path} ({file_size:,} bytes)"
                            )

            # Show preview of results
            workflow_result = results.get("workflow_result")
            if workflow_result and hasattr(workflow_result, "comprehensive_answer"):
                print(f"\n🎯 Research Summary:")
                self.print_separator("-", 50)
                print(
                    workflow_result.comprehensive_answer[:500] + "..."
                    if len(workflow_result.comprehensive_answer) > 500
                    else workflow_result.comprehensive_answer
                )
                self.print_separator("-", 50)

            print(f"\n✨ Research workflow completed successfully!")
            return True

        except Exception as e:
            print(f"\n❌ Research failed: {e}")
            print("You can continue chatting to refine the plan and try again.")
            return False


async def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive chat-based research workflow"
    )
    parser.add_argument(
        "research_question",
        nargs="?",
        default=None,
        help="Initial research question (optional - you can also provide it during chat)",
    )

    args = parser.parse_args()

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return

    interface = ChatResearchInterface()

    # Get research question
    if args.research_question:
        research_question = args.research_question
    else:
        print("🔬 What would you like to research?")
        print("👤 You: ", end="")
        research_question = input().strip()

        if not research_question:
            print("Please provide a research question to get started.")
            return

    # Start the conversation
    await interface.start_conversation(research_question)


if __name__ == "__main__":
    asyncio.run(main())
