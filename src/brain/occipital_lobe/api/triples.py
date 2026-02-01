from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from src.brain.brainstem import get_brain
from src.brain.occipital_lobe.types import TripleCreate, TripleUpdate
from src.brain.infrastructure.mongo_client import get_mongo_client

router = APIRouter(prefix="/api/triples", tags=["triples"])

def get_mongo():
    return get_mongo_client()

@router.get("")
async def get_triples(
    limit: int = Query(50, ge=1, le=500),
    skip: int = Query(0, ge=0),
    subject: Optional[str] = Query(None),
    predicate: Optional[str] = Query(None)
):
    # This endpoint signature in original code was overloaded (one at root, one at root with diff params).
    # In FastAPI, duplicate paths are handled by order, but here the functionality seems split in the original.
    # The original file had two `@app.get("/api/triples")`. The second one overwrote the first in Python routing if defined sequentially?
    # Wait, looking at original file:
    # Lines 395-427 define `get_triples` with subject/predicate filtering.
    # Lines 480-483 define `get_triples` with just limit/skip calling `list_triples`.
    # AND Python dicts (and FastAPI routes) overwrite if path is same. The LAST one defined wins usually.
    # HOWEVER, the first one has more query params. The second one has default params.
    # I should merge them into a single endpoint that delegates appropriately.
    
    mongo = get_mongo()
    brain = await get_brain()
    if not brain or not brain.hippocampus:
        raise HTTPException(status_code=503, detail="Brain initializing")

    if subject or predicate:
        # Use the filter logic from first implementation
        filter_query = {}
        if subject:
            filter_query["subject"] = {"$regex": subject, "$options": "i"}
        if predicate:
            filter_query["predicate"] = predicate
        
        cursor = mongo.knowledge_graph.find(filter_query).sort("last_accessed", -1).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        
        return [
            {
                "id": str(d["_id"]),
                "subject": d.get("subject"),
                "predicate": d.get("predicate"),
                "object": d.get("object"),
                "confidence": d.get("confidence", 0.8),
                "source_memory_id": d.get("source_memory_id"),
                "access_count": d.get("access_count", 0),
                "created_at": d.get("created_at").isoformat() if d.get("created_at") else None,
                "last_accessed": d.get("last_accessed").isoformat() if d.get("last_accessed") else None
            }
            for d in docs
        ]
    else:
        # Use simpler list_triples from brain implementation (matches second implementation in original file)
        # But wait, `list_triples` in `hippocampus` likely does similar finding.
        return await brain.hippocampus.list_triples(limit=limit, skip=skip)

@router.get("/query")
async def query_triples(entity: str, limit: int = 50):
    mongo = get_mongo()
    cursor = mongo.knowledge_graph.find({
        "$or": [
            {"subject": {"$regex": entity, "$options": "i"}},
            {"object": {"$regex": entity, "$options": "i"}}
        ]
    }).limit(limit)
    docs = await cursor.to_list(length=limit)
    return [
        {
            "id": str(d["_id"]),
            "subject": d.get("subject"),
            "predicate": d.get("predicate"),
            "object": d.get("object"),
            "confidence": d.get("confidence", 0.8)
        }
        for d in docs
    ]

@router.get("/{triple_id}")
async def get_triple_by_id(triple_id: str):
    mongo = get_mongo()
    doc = await mongo.knowledge_graph.find_one({"_id": triple_id})
    
    if not doc:
        raise HTTPException(status_code=404, detail="Triple not found")
    
    return {
        "id": str(doc["_id"]),
        "subject": doc.get("subject"),
        "predicate": doc.get("predicate"),
        "object": doc.get("object"),
        "confidence": doc.get("confidence", 0.8),
        "source_memory_id": doc.get("source_memory_id"),
        "access_count": doc.get("access_count", 0),
        "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
        "last_accessed": doc.get("last_accessed").isoformat() if doc.get("last_accessed") else None
    }

@router.post("")
async def create_triple(triple: TripleCreate):
    try:
        brain = await get_brain()
        if not brain or not brain.hippocampus:
            raise HTTPException(status_code=503, detail="Brain initializing")
        triple_id = await brain.hippocampus.add_triple(
            subject=triple.subject,
            predicate=triple.predicate,
            obj=triple.object,
            confidence=triple.confidence,
            source_memory_id=triple.source_memory_id
        )
        return {"id": triple_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{triple_id}")
async def update_triple(triple_id: str, triple: TripleUpdate):
    updates = triple.dict(exclude_unset=True)
    brain = await get_brain()
    if not brain or not brain.hippocampus:
        raise HTTPException(status_code=503, detail="Brain initializing")
    success = await brain.hippocampus.update_triple(triple_id, updates)
    if not success:
        raise HTTPException(status_code=404, detail="Triple not found")
    return {"status": "updated", "id": triple_id}

@router.delete("/{triple_id}")
async def delete_triple(triple_id: str):
    brain = await get_brain()
    if not brain or not brain.hippocampus:
        raise HTTPException(status_code=503, detail="Brain initializing")
    success = await brain.hippocampus.delete_triple(triple_id)
    if not success:
        raise HTTPException(status_code=404, detail="Triple not found")
    return {"status": "deleted", "id": triple_id}


