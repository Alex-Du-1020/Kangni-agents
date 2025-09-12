"""
Memory service for managing agent short-term and long-term memory
"""
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy import desc, and_, or_, func
from sqlalchemy.orm import Session
import logging
import json
import re

from ..models.history import Memory, QueryHistory, MemoryType, MemoryImportance
from ..models.database import get_db_config
from .history_service import HistoryService

logger = logging.getLogger(__name__)


class MemoryService:
    """Service for managing agent memory (short-term and long-term)"""
    
    def __init__(self):
        self.db_config = get_db_config()
        self.history_service = HistoryService()
        # Short-term memory window (last N interactions)
        self.short_term_window = 10
        # Short-term memory expiry (hours)
        self.short_term_expiry_hours = 24
    
    async def create_memory(
        self,
        user_email: str,
        content: str,
        memory_type: MemoryType = MemoryType.LONG_TERM,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        session_id: Optional[str] = None,
        source_query_id: Optional[int] = None,
        related_entities: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None
    ) -> Memory:
        """
        Create a new memory entry
        
        Args:
            user_email: User's email address
            content: Memory content
            memory_type: Type of memory
            importance: Importance level
            session_id: Session identifier
            source_query_id: Source query ID
            related_entities: Related entities (people, projects, etc.)
            tags: Tags for categorization
            embedding: Vector embedding for similarity search
        
        Returns:
            Created Memory object
        """
        try:
            with self.db_config.session_scope() as session:
                # Set expiry for short-term memories
                expires_at = None
                if memory_type == MemoryType.SHORT_TERM:
                    expires_at = datetime.utcnow() + timedelta(hours=self.short_term_expiry_hours)
                
                memory = Memory(
                    user_email=user_email,
                    session_id=session_id,
                    content=content,
                    memory_type=memory_type,
                    importance=importance,
                    source_query_id=source_query_id,
                    related_entities=related_entities,
                    tags=tags,
                    embedding=embedding,
                    expires_at=expires_at
                )
                session.add(memory)
                session.flush()
                memory_id = memory.id
                logger.info(f"Created {memory_type.value} memory with ID: {memory_id}")
                result = type('Memory', (), {'id': memory_id})()
                return result
        except Exception as e:
            logger.error(f"Failed to create memory: {e}")
            raise
    
    async def get_relevant_memories(
        self,
        user_email: str,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        include_expired: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Get relevant memories for a query (both short-term and long-term)
        
        Args:
            user_email: User's email address
            query: Current query to find relevant memories for
            session_id: Session identifier
            limit: Maximum number of memories to return
            include_expired: Whether to include expired memories
        
        Returns:
            Tuple of (short_term_memories, long_term_memories)
        """
        try:
            with self.db_config.session_scope() as session:
                # Get short-term memories (recent session context)
                short_term_query = session.query(Memory).filter(
                    and_(
                        Memory.user_email == user_email,
                        Memory.memory_type == MemoryType.SHORT_TERM
                    )
                )
                
                if session_id:
                    short_term_query = short_term_query.filter(
                        Memory.session_id == session_id
                    )
                
                if not include_expired:
                    short_term_query = short_term_query.filter(
                        or_(
                            Memory.expires_at.is_(None),
                            Memory.expires_at > datetime.utcnow()
                        )
                    )
                
                short_term_memories = short_term_query.order_by(
                    desc(Memory.created_at)
                ).limit(self.short_term_window).all()
                
                # Get long-term memories (important patterns and facts)
                # Search for relevant content based on keywords
                keywords = self._extract_keywords(query)
                
                long_term_query = session.query(Memory).filter(
                    and_(
                        Memory.user_email == user_email,
                        Memory.memory_type.in_([MemoryType.LONG_TERM, MemoryType.SEMANTIC])
                    )
                )
                
                # Filter by keywords in content
                if keywords:
                    keyword_filters = []
                    for keyword in keywords:
                        keyword_filters.append(Memory.content.contains(keyword))
                    long_term_query = long_term_query.filter(or_(*keyword_filters))
                
                # Prioritize by importance and relevance
                long_term_memories = long_term_query.order_by(
                    desc(Memory.importance),
                    desc(Memory.relevance_score),
                    desc(Memory.access_count)
                ).limit(limit).all()
                
                # Update access count and last accessed time
                for memory in short_term_memories + long_term_memories:
                    memory.access_count = (memory.access_count or 0) + 1
                    memory.last_accessed = datetime.utcnow()
                
                session.commit()
                
                # Convert to dictionaries
                short_term_dicts = [m.to_dict() for m in short_term_memories]
                long_term_dicts = [m.to_dict() for m in long_term_memories]
                
                logger.info(f"Retrieved {len(short_term_dicts)} short-term and {len(long_term_dicts)} long-term memories")
                return short_term_dicts, long_term_dicts
        except Exception as e:
            logger.error(f"Failed to get relevant memories: {e}")
            raise
    
    async def extract_and_store_memories(
        self,
        query_id: int,
        user_email: str,
        question: str,
        answer: str,
        session_id: Optional[str] = None,
        feedback_type: Optional[str] = None
    ) -> List[int]:
        """
        Extract important information from a query interaction and store as memories
        
        Args:
            query_id: ID of the query history entry
            user_email: User's email address
            question: User's question
            answer: Agent's answer
            session_id: Session identifier
            feedback_type: User feedback (like/dislike)
        
        Returns:
            List of created memory IDs
        """
        try:
            memories_created = []
            
            # Extract entities and key information
            entities = self._extract_entities(question + " " + answer)
            facts = self._extract_facts(answer)
            
            # Determine importance based on feedback
            importance = MemoryImportance.MEDIUM
            if feedback_type == "like":
                importance = MemoryImportance.HIGH
            elif feedback_type == "dislike":
                importance = MemoryImportance.LOW
            
            # Store short-term memory (conversation context)
            short_memory = await self.create_memory(
                user_email=user_email,
                content=f"Q: {question}\nA: {answer[:500]}",  # Truncate for short-term
                memory_type=MemoryType.SHORT_TERM,
                importance=importance,
                session_id=session_id,
                source_query_id=query_id,
                related_entities=entities[:5],  # Limit entities
                tags=["conversation", "recent"]
            )
            memories_created.append(short_memory.id)
            
            # Store semantic memories (facts and knowledge)
            for fact in facts[:3]:  # Limit to top 3 facts
                if len(fact) > 20:  # Only store substantial facts
                    fact_memory = await self.create_memory(
                        user_email=user_email,
                        content=fact,
                        memory_type=MemoryType.SEMANTIC,
                        importance=importance,
                        session_id=session_id,
                        source_query_id=query_id,
                        related_entities=entities[:3],
                        tags=["fact", "knowledge"]
                    )
                    memories_created.append(fact_memory.id)
            
            # Store episodic memory for important interactions
            if importance in [MemoryImportance.HIGH, MemoryImportance.CRITICAL]:
                episodic_memory = await self.create_memory(
                    user_email=user_email,
                    content=f"Important interaction: {question} -> Key insight: {answer[:200]}",
                    memory_type=MemoryType.EPISODIC,
                    importance=importance,
                    session_id=session_id,
                    source_query_id=query_id,
                    related_entities=entities,
                    tags=["important", "episodic"]
                )
                memories_created.append(episodic_memory.id)
            
            logger.info(f"Created {len(memories_created)} memories from query {query_id}")
            return memories_created
        except Exception as e:
            logger.error(f"Failed to extract and store memories: {e}")
            raise
    
    async def consolidate_memories(
        self,
        user_email: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Consolidate short-term memories into long-term memories
        This should be called periodically or at the end of a session
        
        Args:
            user_email: User's email address
            session_id: Session identifier
        
        Returns:
            Summary of consolidation results
        """
        try:
            with self.db_config.session_scope() as session:
                # Get short-term memories for consolidation
                short_term_query = session.query(Memory).filter(
                    and_(
                        Memory.user_email == user_email,
                        Memory.memory_type == MemoryType.SHORT_TERM,
                        Memory.access_count >= 2  # Accessed at least twice
                    )
                )
                
                if session_id:
                    short_term_query = short_term_query.filter(
                        Memory.session_id == session_id
                    )
                
                memories_to_consolidate = short_term_query.all()
                
                consolidated_count = 0
                for memory in memories_to_consolidate:
                    # Check if this information is important enough
                    if memory.importance in [MemoryImportance.HIGH, MemoryImportance.CRITICAL]:
                        # Convert to long-term memory
                        memory.memory_type = MemoryType.LONG_TERM
                        memory.expires_at = None  # Remove expiry
                        consolidated_count += 1
                
                session.commit()
                
                # Clean up old expired memories
                expired_memories = session.query(Memory).filter(
                    and_(
                        Memory.user_email == user_email,
                        Memory.expires_at < datetime.utcnow()
                    )
                ).delete()
                
                session.commit()
                
                result = {
                    "consolidated": consolidated_count,
                    "expired_deleted": expired_memories,
                    "total_processed": len(memories_to_consolidate)
                }
                
                logger.info(f"Memory consolidation for {user_email}: {result}")
                return result
        except Exception as e:
            logger.error(f"Failed to consolidate memories: {e}")
            raise
    
    async def search_memories(
        self,
        user_email: str,
        search_term: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search memories by content
        
        Args:
            user_email: User's email address
            search_term: Term to search for
            memory_types: Types of memories to search
            limit: Maximum results
        
        Returns:
            List of matching memories
        """
        try:
            with self.db_config.session_scope() as session:
                query = session.query(Memory).filter(
                    and_(
                        Memory.user_email == user_email,
                        Memory.content.contains(search_term)
                    )
                )
                
                if memory_types:
                    query = query.filter(Memory.memory_type.in_(memory_types))
                
                results = query.order_by(
                    desc(Memory.importance),
                    desc(Memory.relevance_score)
                ).limit(limit).all()
                
                memory_dicts = [m.to_dict() for m in results]
                logger.info(f"Found {len(memory_dicts)} memories matching '{search_term}'")
                return memory_dicts
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            raise
    
    async def get_memory_context_for_agent(
        self,
        user_email: str,
        question: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get formatted memory context for the agent to use in reasoning
        
        Args:
            user_email: User's email address
            question: Current question
            session_id: Session identifier
        
        Returns:
            Formatted memory context dictionary
        """
        try:
            # Get relevant memories
            short_term, long_term = await self.get_relevant_memories(
                user_email=user_email,
                query=question,
                session_id=session_id,
                limit=5
            )
            
            # Get recent query history as additional context
            recent_history = await self.history_service.get_user_history(
                user_email=user_email,
                limit=5,
                include_feedback=False,
                include_comments=False
            )
            
            # Format context for agent
            context = {
                "short_term_memories": [
                    {
                        "content": m["content"],
                        "created_at": m["created_at"],
                        "importance": m["importance"]
                    } for m in short_term[:3]  # Limit to avoid context overload
                ],
                "long_term_memories": [
                    {
                        "content": m["content"],
                        "importance": m["importance"],
                        "relevance": m["relevance_score"]
                    } for m in long_term[:3]
                ],
                "recent_interactions": [
                    {
                        "question": h["question"],
                        "answer": h["answer"][:200] if h["answer"] else None,
                        "success": h["success"],
                        "created_at": h["created_at"]
                    } for h in recent_history[:3]
                ],
                "user_profile": {
                    "email": user_email,
                    "total_interactions": len(recent_history),
                    "session_id": session_id
                }
            }
            
            logger.info(f"Prepared memory context for agent with {len(short_term)} short-term and {len(long_term)} long-term memories")
            return context
        except Exception as e:
            logger.error(f"Failed to get memory context for agent: {e}")
            # Return empty context on failure (don't break the agent)
            return {
                "short_term_memories": [],
                "long_term_memories": [],
                "recent_interactions": [],
                "user_profile": {"email": user_email, "session_id": session_id}
            }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from Chinese text for memory search"""
        # Chinese stop words (common function words)
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '个', '对', '下', '过', '他', '她', '它', '们', '我们', '你们', '他们', '她们', '它们', '这个', '那个', '这些', '那些', '什么', '怎么', '为什么', '哪里', '什么时候', '谁', '多少', '几个', '一些', '很多', '非常', '特别', '比较', '更', '最', '还', '又', '再', '已经', '正在', '将要', '可以', '能够', '应该', '必须', '需要', '想要', '希望', '如果', '因为', '所以', '但是', '然后', '而且', '或者', '虽然', '尽管', '无论', '不管', '只要', '除非', '除了', '关于', '对于', '根据', '按照', '通过', '由于', '为了', '以便', '以免', '除非', '除了', '关于', '对于', '根据', '按照', '通过', '由于', '为了', '以便', '以免'
        }
        
        # Extract Chinese characters (2-10 characters) and English words
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,10}', text)
        english_words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Combine and filter
        all_words = chinese_words + english_words
        keywords = list(set([w for w in all_words if w not in stop_words]))
        
        # Sort by length (longer words are often more specific)
        keywords.sort(key=len, reverse=True)
        return keywords[:10]  # Return top 10 keywords
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities (people, places, things) from Chinese text"""
        entities = []
        
        # Look for Chinese proper nouns (2-6 characters, often capitalized or in specific contexts)
        chinese_proper_nouns = re.findall(r'[\u4e00-\u9fff]{2,6}', text)
        entities.extend(chinese_proper_nouns)
        
        # Look for quoted strings (both Chinese and English quotes)
        quoted_chinese = re.findall(r'["""]([^"""]+)["""]', text)
        quoted_english = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted_chinese)
        entities.extend(quoted_english)
        
        # Look for technical terms (English words with special characters)
        technical = re.findall(r'\b\w+[_\-\.]\w+\b', text)
        entities.extend(technical)
        
        # Look for Chinese technical terms (often contain numbers or special characters)
        chinese_technical = re.findall(r'[\u4e00-\u9fff]+[0-9]+[\u4e00-\u9fff]*|[\u4e00-\u9fff]*[0-9]+[\u4e00-\u9fff]+', text)
        entities.extend(chinese_technical)
        
        # Look for English capitalized words (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(capitalized)
        
        # Remove duplicates and return
        return list(set(entities))[:10]
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual statements from Chinese text"""
        facts = []
        
        # Split into sentences (Chinese and English punctuation)
        sentences = re.split(r'[.!?。！？；;]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Look for Chinese declarative patterns
            chinese_fact_patterns = [
                '是', '为', '有', '存在', '包含', '包括', '具有', '拥有', '属于', '位于', '处于',
                '表示', '意味着', '说明', '表明', '显示', '体现', '反映', '代表', '象征',
                '可以', '能够', '会', '将', '要', '应该', '必须', '需要', '要求',
                '根据', '按照', '依据', '基于', '通过', '利用', '使用', '采用',
                '由于', '因为', '所以', '因此', '从而', '导致', '造成', '引起',
                '例如', '比如', '如', '像', '类似', '相似', '相同', '不同', '区别'
            ]
            
            # Look for English declarative patterns
            english_fact_patterns = [' is ', ' are ', ' was ', ' were ', ' has ', ' have ', ' can ', ' will ', ' should ', ' must ']
            
            # Check if sentence contains fact indicators
            has_chinese_fact = any(pattern in sentence for pattern in chinese_fact_patterns)
            has_english_fact = any(pattern in sentence.lower() for pattern in english_fact_patterns)
            
            if has_chinese_fact or has_english_fact:
                # Reasonable length for Chinese (10-100 characters) and English (20-200 characters)
                if 10 <= len(sentence) <= 200:
                    facts.append(sentence)
        
        return facts[:5]  # Return top 5 facts


# Global memory service instance
memory_service = MemoryService()