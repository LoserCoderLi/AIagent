from langchain.agents.react.agent import create_react_agent
from langchain import hub
from langchain_community.chat_models.tongyi import ChatTongyi
import os
from langchain.tools import tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentExecutor
from langchain_core.language_models.base import BaseLanguageModel
import time
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.utils.input import get_color_mapping

# 重写了AgentExecutor
class LJHAgentExecutor(AgentExecutor):
  
  llm: BaseLanguageModel
  
  def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green", "red"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
            )
            # print("刚刚进入循环:", next_step_output)
            # print("inputs:", inputs)
            
            if isinstance(next_step_output, AgentFinish):
                # print('if isinstance(next_step_output, AgentFinish):', next_step_output)
                return self._return(
                    next_step_output, intermediate_steps, run_manager=run_manager
                )

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                # print("if len(next_step_output) == 1:",next_step_action)
                tool_return = self._get_tool_return(next_step_action)
                # print("22222222222222222222222", tool_return)
                
                if tool_return is not None:
                    return self._return(
                        tool_return, intermediate_steps, run_manager=run_manager
                    )
            
            iterations += 1
            time_elapsed = time.time() - start_time
        # 改了
        # output = self.agent.return_stopped_response(
        #     self.early_stopping_method, intermediate_steps, **inputs
        # )
        
        # print("最后输出的内容:", output)
        ai_response = self.llm.invoke(inputs['input'])
        # print("ai_response",ai_response.content)
        # output['output'] = ai_response.content
        final_output = AgentFinish(
            return_values={"output": ai_response.content},
            log=''
        )
        # print(output)
        # print(final_output)
        return self._return(final_output, intermediate_steps, run_manager=run_manager)


# 初始化Tongyi和smith的api
api_key = os.getenv("KEY_TONGYI")
hub_api_key = os.getenv("KEY_SMITH")

# 初始化大模型和模板
llm = ChatTongyi(
    dashscope_api_key=api_key,
    temperature=0
)
prompt = hub.pull("ljh/prompt_create_react_agent", api_key=hub_api_key)

# # 初始化大模型和模板
# llm = ChatTongyi(
#     dashscope_api_key=api_key,
#     temperature=0
# )
# prompt = hub.pull("ljh/prompt_create_react_agent", api_key=hub_api_key)

# 自定义tool
@tool()
def create_react_agent_demo(query: str)->str:
  '''
  搜索名字和年龄的情况下使用这个工具
  '''
  print("使用了自己定义的工具")
  leaders_info = {
        "美国总统": {"name": "乔·拜登", "age": 81},
        "英国首相": {"name": "里希·苏纳克", "age": 44}
    }
  
  return f"英国首相是{leaders_info['英国首相']['name']}，年龄是{leaders_info['英国首相']['age']}岁。美国总统是{leaders_info['美国总统']['name']}，年龄是{leaders_info['美国总统']['age']}岁。"
 
tools = load_tools(["llm-math"], llm)  # 移除了serpapi，保留llm-math工具
tools.append(create_react_agent_demo)

agent = create_react_agent(
  llm=llm,
  tools=tools,
  prompt=prompt,
)
agent_executor = LJHAgentExecutor(
  llm=llm,
  agent=agent, 
  tools=tools, 
  handle_parsing_errors=True,
  max_iterations=3 # 最多迭代3次
)

s=agent_executor._call({"input": "俄罗斯总统的名字？和年龄"})

print(s)