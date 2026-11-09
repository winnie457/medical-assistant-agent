"""对话 API 路由"""

import json
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ...models.schemas import ChatRequest
from ...agents.medical_agent import get_medical_agent
from ...services.conversation_service import get_conversation_manager
from ...services.memory_service import update_memory_after_chat
from ...services.suspension_service import get_suspension_manager
from ...memory.memory_manager import get_memory_manager

router = APIRouter(prefix="/chat", tags=["对话"])


# ============ HITL 请求模型 ============

class ApprovalRequest(BaseModel):
    conversation_id: str


class RejectRequest(BaseModel):
    conversation_id: str
    reason: str = "用户取消"


@router.post(
    "",
    summary="发送消息（非流式）",
    description="向智能就医助手发送消息，获取完整回答"
)
async def chat(request: ChatRequest):
    """非流式对话接口"""
    try:
        agent = get_medical_agent()
        conv_manager = get_conversation_manager()

        # 获取或创建会话
        conversation_id = request.conversation_id
        user_id = request.user_id or "default"
        if not conversation_id:
            conversation_id = conv_manager.create_conversation()

        # 获取历史消息
        history = conv_manager.get_langchain_messages(conversation_id)

        # 调用 Agent（graph.invoke 为同步阻塞实现，放入线程池避免阻塞事件循环）
        result = await asyncio.to_thread(
            agent.chat,
            message=request.message,
            history=history,
            city=request.city or "北京",
            user_id=user_id,
            conversation_id=conversation_id,
        )

        # 保存用户消息和助手回答
        conv_manager.add_message(conversation_id, "user", request.message)
        conv_manager.add_message(conversation_id, "assistant", result["answer"])

        # 【后台任务】非流式入口同样触发长期记忆抽取，保持与流式一致
        asyncio.create_task(update_memory_after_chat(
            user_id=user_id,
            conversation_id=conversation_id,
            conversation_text=f"用户: {request.message}\n助手: {result['answer']}",
        ))

        # 非流式入口没有交互通道：高风险工具会被 fail-closed 拦截，
        # 这里回传标记，提示调用方改用流式接口完成用户确认
        blocked = [
            h for h in result["tool_calls_history"]
            if h.get("status") == "blocked_requires_approval"
        ]

        return {
            "success": True,
            "conversation_id": conversation_id,
            "answer": result["answer"],
            "tool_calls_history": result["tool_calls_history"],
            "iteration_count": result["iteration_count"],
            "requires_approval": bool(blocked),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"对话失败: {str(e)}"
        )


@router.post(
    "/stream",
    summary="发送消息（流式）",
    description="流式对话接口，逐步返回思考过程、工具调用和最终回答"
)
async def chat_stream(request: ChatRequest):
    """流式对话接口 - SSE 方式"""
    try:
        agent = get_medical_agent()
        conv_manager = get_conversation_manager()

        # 获取或创建会话
        conversation_id = request.conversation_id
        user_id = request.user_id or "default"
        if not conversation_id:
            conversation_id = conv_manager.create_conversation()

        # 获取历史消息
        history = conv_manager.get_langchain_messages(conversation_id)

        # 保存用户消息
        conv_manager.add_message(conversation_id, "user", request.message)

        async def event_generator():
            full_answer = ""
            tool_calls_history = []
            conversation_text = f"用户：{request.message}\n"  # 收集对话文本用于记忆提取

            async for event in agent.chat_stream(
                message=request.message,
                history=history,
                city=request.city or "北京",
                conversation_id=conversation_id,
                user_id=user_id,
            ):
                event_type = event.get("type", "")

                if event_type == "thought":
                    yield f"data: {json.dumps({'type': 'thought', 'content': event.get('content', ''), 'iteration': event.get('iteration', 0)}, ensure_ascii=False)}\n\n"

                elif event_type == "tool_call":
                    tool_info = {
                        "type": "tool_call",
                        "tool_name": event.get("tool_name", ""),
                        "tool_description": event.get("tool_description", ""),
                        "arguments": event.get("arguments", {}),
                        "risk_level": event.get("risk_level", "low"),
                        "risk_label": event.get("risk_label", ""),
                        "iteration": event.get("iteration", 0)
                    }
                    yield f"data: {json.dumps(tool_info, ensure_ascii=False)}\n\n"

                elif event_type == "approval_required":
                    # 高风险工具需要用户确认
                    approval_info = {
                        "type": "approval_required",
                        "conversation_id": conversation_id,
                        "tool_name": event.get("tool_name", ""),
                        "tool_args": event.get("tool_args", {}),
                        "risk_level": event.get("risk_level", "high"),
                        "title": event.get("title", "确认操作"),
                        "message": event.get("message", ""),
                        "warning": event.get("warning", ""),
                        "confirm_text": event.get("confirm_text", "确认"),
                        "cancel_text": event.get("cancel_text", "取消"),
                        "iteration": event.get("iteration", 0),
                    }
                    yield f"data: {json.dumps(approval_info, ensure_ascii=False)}\n\n"

                elif event_type == "tool_result":
                    result_info = {
                        "type": "tool_result",
                        "tool_name": event.get("tool_name", ""),
                        "status": event.get("status", ""),
                        "risk_level": event.get("risk_level", "low"),
                        "retry_count": event.get("retry_count", 0),
                        "used_fallback": event.get("used_fallback", False),
                        "result_preview": event.get("result_preview", ""),
                        "duration": event.get("duration", 0),
                        "iteration": event.get("iteration", 0)
                    }
                    tool_calls_history.append({
                        "tool_name": event.get("tool_name", ""),
                        "status": event.get("status", ""),
                        "duration": event.get("duration", 0)
                    })
                    yield f"data: {json.dumps(result_info, ensure_ascii=False)}\n\n"

                elif event_type == "answer_delta":
                    # token 级增量：前端追加显示，实现打字机效果
                    yield f"data: {json.dumps({'type': 'answer_delta', 'content': event.get('content', '')}, ensure_ascii=False)}\n\n"

                elif event_type == "answer_reset":
                    # 模型转入工具调用，清空已推送的正文增量
                    yield 'data: {"type": "answer_reset"}\n\n'

                elif event_type == "answer":
                    full_answer = event.get("content", "")
                    conversation_text += f"助手：{full_answer}\n"
                    # 流式输出最终答案
                    # 这里简化为一次性输出，也可以做真正的逐字输出
                    answer_event = {
                        "type": "answer",
                        "content": full_answer,
                        "done": event.get("done", False)
                    }
                    yield f"data: {json.dumps(answer_event, ensure_ascii=False)}\n\n"

                elif event_type == "done":
                    # 保存助手回答
                    conv_manager.add_message(conversation_id, "assistant", full_answer)

                    done_event = {
                        "type": "done",
                        "conversation_id": conversation_id,
                        "iteration_count": event.get("iteration_count", 0),
                        "tool_calls_count": event.get("tool_calls_count", 0),
                        "tool_calls_history": event.get("tool_calls_history", [])
                    }
                    yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"
                    
                    # 【后台任务】从对话中提取健康信息，更新用户画像
                    # 用 create_task 后台运行，不阻塞流式响应
                    asyncio.create_task(update_memory_after_chat(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        conversation_text=conversation_text,
                    ))

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"流式对话失败: {str(e)}"
        )


@router.get(
    "/conversations",
    summary="获取会话列表",
    description="获取所有历史会话列表"
)
async def list_conversations():
    """获取会话列表"""
    try:
        conv_manager = get_conversation_manager()
        conversations = conv_manager.list_conversations()
        return {
            "success": True,
            "data": conversations,
            "total": len(conversations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/conversations/{conversation_id}",
    summary="获取会话详情",
    description="获取指定会话的所有消息"
)
async def get_conversation(conversation_id: str):
    """获取会话详情"""
    try:
        conv_manager = get_conversation_manager()
        conv = conv_manager.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="会话不存在")

        return {
            "success": True,
            "data": {
                "conversation_id": conv["conversation_id"],
                "title": conv["title"],
                "created_at": conv["created_at"],
                "messages": conv["messages"]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ 用户画像（长期记忆）接口 ============

@router.get(
    "/profile",
    summary="获取用户健康画像",
    description="获取指定用户的健康画像（长期记忆）"
)
async def get_user_profile(user_id: str = "default"):
    """获取用户健康画像"""
    try:
        memory_manager = get_memory_manager()
        profile = memory_manager.get_profile_summary(user_id)
        return {
            "success": True,
            "data": profile
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/profile/memories/{memory_id}",
    summary="删除一条记忆",
    description="软删除指定的记忆条目"
)
async def delete_memory(memory_id: str, user_id: str = "default"):
    """删除一条记忆"""
    try:
        memory_manager = get_memory_manager()
        success = memory_manager.delete_memory(user_id, memory_id)
        if not success:
            raise HTTPException(status_code=404, detail="记忆不存在")
        return {"success": True, "message": "记忆已删除"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/profile/memories",
    summary="清空所有记忆",
    description="软删除用户的所有健康记忆"
)
async def clear_all_memories(user_id: str = "default"):
    """清空所有记忆"""
    try:
        memory_manager = get_memory_manager()
        count = memory_manager.clear_all_memories(user_id)
        return {"success": True, "message": f"已清空 {count} 条记忆", "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ HITL 风险确认接口 ============

@router.post(
    "/approve",
    summary="确认执行高风险工具",
    description="用户确认执行高风险医疗工具（如用药咨询），Agent 继续执行"
)
async def approve_tool(request: ApprovalRequest):
    """用户确认执行高风险工具"""
    try:
        suspension_manager = get_suspension_manager()
        success = await suspension_manager.approve(request.conversation_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="会话不在等待确认状态，可能已超时或已处理"
            )
        
        return {"success": True, "message": "已确认执行"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/reject",
    summary="拒绝执行高风险工具",
    description="用户拒绝执行高风险医疗工具，Agent 将收到拒绝信息并调整策略"
)
async def reject_tool(request: RejectRequest):
    """用户拒绝执行高风险工具"""
    try:
        suspension_manager = get_suspension_manager()
        success = await suspension_manager.reject(request.conversation_id, request.reason)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="会话不在等待确认状态，可能已超时或已处理"
            )
        
        return {"success": True, "message": "已拒绝执行"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/conversations/{conversation_id}",
    summary="删除会话",
    description="删除指定的会话"
)
async def delete_conversation(conversation_id: str):
    """删除会话"""
    try:
        conv_manager = get_conversation_manager()
        success = conv_manager.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail="会话不存在")

        return {"success": True, "message": "会话已删除"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
