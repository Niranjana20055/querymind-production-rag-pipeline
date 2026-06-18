from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()
engine = create_engine("sqlite:///data/query_logs.db")

class QueryLog(Base):
    __tablename__ = "query_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(String)
    answer = Column(String)
    faithfulness = Column(Float, default=0.0)
    answer_relevance = Column(Float, default=0.0)
    context_precision = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.now)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def log_query(question, answer, scores):
    session = Session()
    log = QueryLog(
        question=question,
        answer=answer,
        faithfulness=scores.get("faithfulness", 0.0),
        answer_relevance=scores.get("answer_relevance", 0.0),
        context_precision=scores.get("context_precision", 0.0)
    )
    session.add(log)
    session.commit()
    session.close()

def get_all_logs():
    session = Session()
    logs = session.query(QueryLog).all()
    logs_data = [
        {
            "question": l.question,
            "answer": l.answer,
            "faithfulness": l.faithfulness,
            "answer_relevance": l.answer_relevance,
            "context_precision": l.context_precision,
            "timestamp": l.timestamp
        }
        for l in logs
    ]
    session.close()
    return logs_data