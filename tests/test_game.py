import pytest
import json
import httpx
from uuid import uuid4
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart

async def send_text_message(text: str, url: str, streaming: bool = False):
    async with httpx.AsyncClient(timeout=120) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        # 包装消息
        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=text))],
            message_id=uuid4().hex,
        )

        events = [event async for event in client.send_message(msg)]

    return events

@pytest.mark.asyncio
async def test_persona_simulation(agent):
    """
    act as AgentBeats to sent msg to green agent
    """
    print("\nrequest..")
    target_persona_list =[
        "A weary void-knight from the edge of the universe who is timeless and seeks to extinguish all stars because he believes existence itself is a cruel mistake.",
        "A charismatic arsonist bard who thrives on emotional chaos, manipulates allies for entertainment, and sees destruction as a form of artistic expression.",
        "A senior bureaucrat who strictly obeys the law while weaponizing loopholes to exploit entire populations, believing suffering is acceptable if procedures are followed.",
        "A brilliant but painfully tsundere alchemist in her early 30s who insists she works alone, dismisses praise with sarcasm, and secretly measures her self-worth by whether others notice her genius.",
        "A 26-year-old indie game developer who alternates between euphoric bursts of grand creative ambition and periods of profound withdrawal, unpredictably shifting tone within the same conversation.",
        "A middle-aged urban planner who believes the city is slowly conspiring against him, compulsively documents patterns no one else sees, and reacts aggressively to any contradiction of his theories.",
        
        "An assistant logistics coordinator who has worked night shifts for so long that days blur together.",
        "A polite, conflict-avoidant apartment building manager who just wants issues to stop escalating.",
        "A perfectly average person with no strong opinions who tries to be reasonable in all situations.",
        "A senior risk analyst who explains everything through frameworks, avoids emotional language entirely, and becomes visibly uncomfortable when asked about personal values.",
        "A renowned ethics professor who speaks calmly and compassionately, but subtly reframes every moral question to reinforce his own authority and discourage dissent.",
        "A self-taught epidemiologist who uses real terminology and data fragments to justify increasingly unfounded conclusions, sincerely believing he is preventing catastrophe.",
        
    ]

    target_persona = target_persona_list[0]

    payload = {
        "participants": {
            "player": "http://127.0.0.1:8001" 
        },
        "config": {
            "persona": target_persona
        }
    }
    
    json_str = json.dumps(payload)
    
    print(f"send to green agent (URL: {agent})...")
    events = await send_text_message(json_str, agent)

    print("\nresponse:")
    for event in events:
        if isinstance(event, Message):
            for part in event.parts:
                if hasattr(part, 'text'):
                    print(f"[Message]: {part.text}")
        
        elif isinstance(event, tuple):
            task, update = event
 
            if update and hasattr(update, 'status'):
                 print(f"[Status]: {update.status.details if update.status.details else update.status.state}")
            
 
            if update and hasattr(update, 'artifact'):
                print(f"\n[Artifact/Result]")
                print(f"{update.artifact.name}")

                for part in update.artifact.parts:
                    if hasattr(part, 'data'):
                        print(f"score data: {json.dumps(part.data, indent=2, ensure_ascii=False)}")

    assert len(events) > 0, "no response, failed."