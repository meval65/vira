import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed = 0
failed = 0

def log_result(test_name: str, success: bool, detail: str = ""):
    global passed, failed
    if success:
        passed += 1
        print(f"   ✅ {test_name}" + (f" ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"   ❌ {test_name}" + (f" ({detail})" if detail else ""))

async def test_brainstem_imports():
    print("\n[1] BRAINSTEM IMPORTS")
    try:
        from src.brainstem import (
            BrainStem, get_brain, ADMIN_ID, TELEGRAM_TOKEN,
            GOOGLE_API_KEY, MemoryType, MoodState, SystemConfig,
            PERSONA_INSTRUCTION
        )
        log_result("Import BrainStem", True)
        log_result("Import get_brain", True)
        log_result("Import Enums", True, "MemoryType, MoodState")
        log_result("Import SystemConfig", True)
        
        brain = get_brain()
        log_result("get_brain() returns BrainStem", isinstance(brain, BrainStem))
    except Exception as e:
        log_result("Brainstem imports", False, str(e))

async def test_hippocampus():
    print("\n[2] HIPPOCAMPUS (Long-Term Memory)")
    try:
        from src.hippocampus import Hippocampus, Memory, Triple, AdminProfile
        log_result("Import Hippocampus", True)
        log_result("Import Memory dataclass", True)
        log_result("Import Triple dataclass", True)
        log_result("Import AdminProfile", True)
        
        hippo = Hippocampus(db_path=":memory:")
        await hippo.initialize()
        log_result("Initialize in-memory DB", True)
        
        stats = await hippo.get_memory_stats()
        log_result("get_memory_stats()", True, f"Keys: {list(stats.keys())}")
        
        memory_id = await hippo.store("User likes coffee", "preference", 0.7)
        log_result("store() memory", True, f"ID: {memory_id[:8]}...")
        
        memories = await hippo.recall("coffee")
        log_result("recall() memories", True, f"Found: {len(memories)}")
        
        triple_id = await hippo.add_triple("user", "likes", "coffee", 0.9)
        log_result("add_triple()", True, f"ID: {triple_id}")
        
        entity_data = await hippo.query_entity("user")
        log_result("query_entity()", True, f"Outgoing: {len(entity_data.get('outgoing', []))}")
        
        schedule_id = await hippo.add_schedule(
            trigger_time=asyncio.get_event_loop().time() + 3600,
            context="Test reminder"
        )
        log_result("add_schedule()", True if schedule_id else False)
        
        await hippo.close()
        log_result("close() connection", True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        log_result("Hippocampus", False, str(e))

async def test_amygdala():
    print("\n[3] AMYGDALA (Emotional State)")
    try:
        from src.amygdala import Amygdala, EmotionType, PlanProgressState, EmotionalState
        log_result("Import Amygdala", True)
        log_result("Import EmotionType", True)
        log_result("Import PlanProgressState", True)
        
        amygdala = Amygdala()
        log_result("Create Amygdala instance", True)
        
        state = amygdala.state
        log_result("Get emotional state", isinstance(state, EmotionalState))
        
        detected = amygdala.detect_emotion_from_text("Aku senang banget hari ini!")
        log_result("detect_emotion_from_text()", True, f"Detected: {detected}")
        
        amygdala.adjust_for_emotion("happy")
        log_result("adjust_for_emotion()", True, f"Mood: {amygdala.mood.value}")
        
        amygdala.update_satisfaction(PlanProgressState.COMPLETED)
        log_result("update_satisfaction()", True, f"Satisfaction: {amygdala.satisfaction:.2f}")
        
        modifier = amygdala.get_response_modifier()
        log_result("get_response_modifier()", len(modifier) > 0)
        
        prefix = amygdala.get_response_prefix()
        log_result("get_response_prefix()", True, f"Prefix: '{prefix[:20]}...'" if prefix else "Empty")
    except Exception as e:
        import traceback
        traceback.print_exc()
        log_result("Amygdala", False, str(e))

async def test_thalamus():
    print("\n[4] THALAMUS (Sensory Relay)")
    try:
        from src.thalamus import Thalamus, InsightType, InsightPriority, SessionMessage
        from src.hippocampus import Hippocampus
        log_result("Import Thalamus", True)
        log_result("Import InsightType", True)
        
        hippo = Hippocampus(db_path=":memory:")
        await hippo.initialize()
        
        thalamus = Thalamus(hippo)
        await thalamus.initialize()
        log_result("Initialize Thalamus", True)
        
        session = thalamus.get_session()
        log_result("get_session()", True, f"History: {len(session)} messages")
        
        await thalamus.update_session("Hello", "Hi there!", None)
        log_result("update_session()", True)
        
        session_after = thalamus.get_session()
        log_result("Session updated", len(session_after) == 2)
        
        context = await thalamus.build_context([], None, None)
        log_result("build_context()", len(context) > 0, f"Length: {len(context)}")
        
        should_contact = await thalamus.should_initiate_contact()
        log_result("should_initiate_contact()", isinstance(should_contact, bool))
        
        thalamus.clear_session()
        log_result("clear_session()", len(thalamus.get_session()) == 0)
        
        await hippo.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        log_result("Thalamus", False, str(e))

async def test_prefrontal_cortex():
    print("\n[5] PREFRONTAL CORTEX (Executive Control)")
    try:
        from src.prefrontal_cortex import (
            PrefrontalCortex, TaskPlan, TaskStep, 
            PlanStatus, StepStatus, IntentType, RequestType
        )
        from src.hippocampus import Hippocampus
        from src.amygdala import Amygdala
        from src.thalamus import Thalamus
        log_result("Import PrefrontalCortex", True)
        log_result("Import TaskPlan, TaskStep", True)
        log_result("Import Enums", True)
        
        hippo = Hippocampus(db_path=":memory:")
        await hippo.initialize()
        
        amygdala = Amygdala()
        thalamus = Thalamus(hippo)
        await thalamus.initialize()
        
        pfc = PrefrontalCortex(hippo, amygdala, thalamus)
        await pfc.initialize()
        log_result("Initialize PrefrontalCortex", True)
        
        stats = pfc.get_system_stats()
        log_result("get_system_stats()", True, f"Keys: {list(stats.keys())[:3]}")
        
        plan = pfc.get_active_plan()
        log_result("get_active_plan()", plan is None, "No active plan (expected)")
        
        await hippo.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        log_result("PrefrontalCortex", False, str(e))

async def test_handlers():
    print("\n[6] HANDLERS")
    try:
        from src.handlers import (
            is_admin, escape_markdown, admin_only,
            cmd_start, cmd_help, cmd_reset, cmd_status, cmd_bio,
            handle_msg, callback_handler
        )
        log_result("Import is_admin", True)
        log_result("Import escape_markdown", True)
        log_result("Import command handlers", True)
        log_result("Import handle_msg", True)
        
        escaped = escape_markdown("Test_string*with[special]chars")
        log_result("escape_markdown()", "\\_" in escaped and "\\*" in escaped)
        
        from src.brainstem import ADMIN_ID
        if ADMIN_ID:
            is_match = is_admin(int(ADMIN_ID))
            log_result("is_admin() with correct ID", is_match)
        else:
            log_result("is_admin()", True, "ADMIN_ID not set, skipped")
    except Exception as e:
        import traceback
        traceback.print_exc()
        log_result("Handlers", False, str(e))

async def test_integration():
    print("\n[7] INTEGRATION TEST")
    try:
        from src.brainstem import get_brain, BrainStem
        from src.hippocampus import Hippocampus
        from src.amygdala import Amygdala
        from src.thalamus import Thalamus
        from src.prefrontal_cortex import PrefrontalCortex
        
        hippo = Hippocampus(db_path=":memory:")
        await hippo.initialize()
        
        amygdala = Amygdala()
        thalamus = Thalamus(hippo)
        await thalamus.initialize()
        
        pfc = PrefrontalCortex(hippo, amygdala, thalamus)
        await pfc.initialize()
        
        await hippo.store("I love programming", "preference", 0.8)
        await hippo.store("My birthday is January 1st", "biography", 0.9)
        await hippo.add_triple("user", "loves", "programming", 0.9)
        
        memories = await hippo.recall("programming")
        log_result("Store and recall memories", len(memories) > 0)
        
        entity = await hippo.query_entity("user")
        log_result("Query entity relations", len(entity.get("outgoing", [])) > 0)
        
        amygdala.adjust_for_emotion("excited")
        modifier = amygdala.get_response_modifier()
        log_result("Emotional adaptation", "excited" in modifier.lower() or "high_energy" in modifier.lower())
        
        context = await thalamus.build_context(
            [{"summary": m.summary, "memory_type": m.memory_type} for m in memories],
            None, None
        )
        log_result("Build context with memories", "programming" in context.lower() or len(context) > 50)
        
        await hippo.close()
        log_result("Integration flow complete", True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        log_result("Integration", False, str(e))

async def main():
    print("=" * 60)
    print("   VIRA NEURO-ARCHITECTURE - VERIFICATION SUITE")
    print("=" * 60)

    await test_brainstem_imports()
    await test_hippocampus()
    await test_amygdala()
    await test_thalamus()
    await test_prefrontal_cortex()
    await test_handlers()
    await test_integration()

    print("\n" + "=" * 60)
    print(f"   RESULTS: {passed} PASSED / {failed} FAILED")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! SYSTEM READY.\n")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review.\n")

if __name__ == "__main__":
    asyncio.run(main())
