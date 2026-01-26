import os
import logging
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import requests
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from config import Config

logger = logging.getLogger(__name__)

_connection_settings = None
_connection_lock = threading.Lock()

QA_LOG_HEADERS = [
    "timestamp",
    "ta_id",
    "ta_slug",
    "ta_name",
    "course_name",
    "session_id",
    "query",
    "answer",
    "sources",
    "chunk_count",
    "latency_ms",
    "retrieval_latency_ms",
    "generation_latency_ms",
    "token_count",
    "total_chunks_in_ta",
    "filters_applied",
    "filter_match_count",
    "retrieval_method",
    "is_conceptual",
    "score_top1",
    "score_top8",
    "score_mean",
    "score_spread",
    "chunk_scores",
    "chunk_sources_detail",
    "rerank_applied",
    "rerank_method",
    "rerank_latency_ms",
    "llm_score_top1",
    "llm_score_top8",
    "vector_score_top1",
    "top_reasons",
    "pre_rerank_candidates",
    "hybrid_fallback_triggered",
    "hybrid_fallback_reason",
    "hybrid_doc_filename",
    "hybrid_doc_tokens",
    "hybrid_doc_id_method"
]

INDEX_LOG_HEADERS = [
    "timestamp",
    "ta_id",
    "ta_slug",
    "file_name",
    "doc_type",
    "total_pages",
    "raw_text_length",
    "chunk_index",
    "total_chunks",
    "chunk_text_length",
    "chunk_context",
    "chunk_text_preview",
    "enriched_text_preview",
    "has_embedding",
    "status",
    "error_message",
    "headers_found"
]

INDEX_LOG_TAB_NAME = "index_logs"

def _get_access_token() -> Optional[str]:
    global _connection_settings
    
    with _connection_lock:
        if (_connection_settings and 
            _connection_settings.get('settings', {}).get('expires_at') and
            datetime.fromisoformat(_connection_settings['settings']['expires_at'].replace('Z', '+00:00')) > datetime.now(timezone.utc)):
            return _connection_settings['settings'].get('access_token')
    
    hostname = os.environ.get('REPLIT_CONNECTORS_HOSTNAME')
    repl_identity = os.environ.get('REPL_IDENTITY')
    web_repl_renewal = os.environ.get('WEB_REPL_RENEWAL')
    
    if repl_identity:
        x_replit_token = f'repl {repl_identity}'
    elif web_repl_renewal:
        x_replit_token = f'depl {web_repl_renewal}'
    else:
        logger.warning("No Replit token found for Google Sheets authentication")
        return None
    
    if not hostname:
        logger.warning("REPLIT_CONNECTORS_HOSTNAME not set")
        return None
    
    try:
        response = requests.get(
            f'https://{hostname}/api/v2/connection?include_secrets=true&connector_names=google-sheet',
            headers={
                'Accept': 'application/json',
                'X_REPLIT_TOKEN': x_replit_token
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        with _connection_lock:
            _connection_settings = data.get('items', [{}])[0] if data.get('items') else {}
        
        settings = _connection_settings.get('settings', {})
        access_token = settings.get('access_token') or settings.get('oauth', {}).get('credentials', {}).get('access_token')
        
        if not access_token:
            logger.warning("No access token found in Google Sheets connection")
            return None
        
        return access_token
        
    except Exception as e:
        logger.error(f"Failed to get Google Sheets access token: {e}")
        return None

def _get_sheets_service():
    access_token = _get_access_token()
    if not access_token:
        return None
    
    credentials = Credentials(token=access_token)
    return build('sheets', 'v4', credentials=credentials, cache_discovery=False)

def _ensure_headers_exist(service, spreadsheet_id: str, tab_name: str) -> bool:
    num_cols = len(QA_LOG_HEADERS)
    end_col = chr(ord('A') + (num_cols - 1) % 26)
    if num_cols > 26:
        end_col = chr(ord('A') + (num_cols - 1) // 26 - 1) + chr(ord('A') + (num_cols - 1) % 26)
    header_range = f'{tab_name}!A1:{end_col}1'
    
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=header_range
        ).execute()
        
        values = result.get('values', [])
        if not values or values[0] != QA_LOG_HEADERS:
            service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=header_range,
                valueInputOption='RAW',
                body={'values': [QA_LOG_HEADERS]}
            ).execute()
            logger.info(f"Created/updated headers in {tab_name} ({num_cols} columns)")
        
        return True
        
    except Exception as e:
        if 'Unable to parse range' in str(e) or 'not found' in str(e).lower():
            try:
                service.spreadsheets().batchUpdate(
                    spreadsheetId=spreadsheet_id,
                    body={
                        'requests': [{
                            'addSheet': {
                                'properties': {'title': tab_name}
                            }
                        }]
                    }
                ).execute()
                logger.info(f"Created new tab: {tab_name}")
                
                service.spreadsheets().values().update(
                    spreadsheetId=spreadsheet_id,
                    range=header_range,
                    valueInputOption='RAW',
                    body={'values': [QA_LOG_HEADERS]}
                ).execute()
                return True
                
            except Exception as create_error:
                logger.error(f"Failed to create tab {tab_name}: {create_error}")
                return False
        else:
            logger.error(f"Failed to check/create headers: {e}")
            return False

def log_qa_entry(
    ta_id: str,
    ta_slug: str,
    ta_name: str,
    course_name: str,
    session_id: str,
    query: str,
    answer: str,
    sources: list,
    chunk_count: int,
    latency_ms: int,
    retrieval_latency_ms: int,
    generation_latency_ms: int,
    token_count: int,
    retrieval_diagnostics: Optional[Dict[str, Any]] = None
) -> bool:
    if not Config.QA_LOG_SHEET_ID:
        logger.debug("QA logging disabled - no sheet ID configured")
        return False
    
    diag = retrieval_diagnostics or {}
    
    def _do_log():
        try:
            service = _get_sheets_service()
            if not service:
                logger.warning("Could not get Google Sheets service")
                return
            
            if not _ensure_headers_exist(service, Config.QA_LOG_SHEET_ID, Config.QA_LOG_TAB_NAME):
                return
            
            import json
            
            rerank_info = diag.get("rerank_info", {})
            
            pre_rerank = diag.get("pre_rerank_candidates", [])
            pre_rerank_str = json.dumps(pre_rerank) if pre_rerank else "[]"
            
            row = [
                datetime.utcnow().isoformat() + 'Z',
                str(ta_id),
                ta_slug,
                ta_name,
                course_name,
                session_id,
                query[:5000] if query else "",
                answer[:10000] if answer else "",
                ", ".join(sources) if sources else "",
                str(chunk_count),
                str(latency_ms),
                str(retrieval_latency_ms),
                str(generation_latency_ms),
                str(token_count),
                str(diag.get("total_chunks_in_ta", "")),
                diag.get("filters_applied") or "",
                str(diag.get("filter_match_count", "")),
                diag.get("retrieval_method", ""),
                str(diag.get("is_conceptual", "")),
                str(diag.get("score_top1", "")),
                str(diag.get("score_top8", "")),
                str(diag.get("score_mean", "")),
                str(diag.get("score_spread", "")),
                json.dumps(diag.get("chunk_scores", [])),
                json.dumps(diag.get("chunk_sources_detail", [])),
                str(diag.get("rerank_applied", "")),
                str(rerank_info.get("method", "")),
                str(rerank_info.get("rerank_latency_ms", "")),
                str(rerank_info.get("llm_score_top1", "")),
                str(rerank_info.get("llm_score_top8", "")),
                str(rerank_info.get("vector_score_top1", "")),
                json.dumps(rerank_info.get("top_reasons", [])),
                pre_rerank_str[:30000],
                str(diag.get("hybrid_fallback_triggered", False)),
                diag.get("hybrid_fallback_reason") or "",
                diag.get("hybrid_doc_filename") or "",
                str(diag.get("hybrid_doc_tokens", 0)),
                diag.get("hybrid_doc_id_method") or ""
            ]
            
            service.spreadsheets().values().append(
                spreadsheetId=Config.QA_LOG_SHEET_ID,
                range=f'{Config.QA_LOG_TAB_NAME}!A:A',
                valueInputOption='RAW',
                insertDataOption='INSERT_ROWS',
                body={'values': [row]}
            ).execute()
            
            logger.info(f"Logged QA entry for query: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to log QA entry: {e}")
    
    thread = threading.Thread(target=_do_log, daemon=True)
    thread.start()
    return True

def test_connection() -> Dict[str, Any]:
    result = {
        "success": False,
        "message": "",
        "sheet_id": Config.QA_LOG_SHEET_ID,
        "tab_name": Config.QA_LOG_TAB_NAME
    }
    
    if not Config.QA_LOG_SHEET_ID:
        result["message"] = "No sheet ID configured"
        return result
    
    try:
        service = _get_sheets_service()
        if not service:
            result["message"] = "Could not authenticate with Google Sheets"
            return result
        
        spreadsheet = service.spreadsheets().get(
            spreadsheetId=Config.QA_LOG_SHEET_ID
        ).execute()
        
        result["success"] = True
        result["message"] = f"Connected to: {spreadsheet.get('properties', {}).get('title', 'Unknown')}"
        result["spreadsheet_title"] = spreadsheet.get('properties', {}).get('title')
        
        if _ensure_headers_exist(service, Config.QA_LOG_SHEET_ID, Config.QA_LOG_TAB_NAME):
            result["headers_ready"] = True
        
        return result
        
    except Exception as e:
        result["message"] = f"Connection failed: {str(e)}"
        return result


def _ensure_index_headers_exist(service, spreadsheet_id: str) -> bool:
    """Ensure index_logs tab exists with correct headers."""
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=f'{INDEX_LOG_TAB_NAME}!A1:P1'
        ).execute()
        
        values = result.get('values', [])
        if not values or values[0] != INDEX_LOG_HEADERS:
            service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=f'{INDEX_LOG_TAB_NAME}!A1:Q1',
                valueInputOption='RAW',
                body={'values': [INDEX_LOG_HEADERS]}
            ).execute()
            logger.info(f"Created/updated headers in {INDEX_LOG_TAB_NAME}")
        
        return True
        
    except Exception as e:
        if 'Unable to parse range' in str(e) or 'not found' in str(e).lower():
            try:
                service.spreadsheets().batchUpdate(
                    spreadsheetId=spreadsheet_id,
                    body={
                        'requests': [{
                            'addSheet': {
                                'properties': {'title': INDEX_LOG_TAB_NAME}
                            }
                        }]
                    }
                ).execute()
                logger.info(f"Created new tab: {INDEX_LOG_TAB_NAME}")
                
                service.spreadsheets().values().update(
                    spreadsheetId=spreadsheet_id,
                    range=f'{INDEX_LOG_TAB_NAME}!A1:P1',
                    valueInputOption='RAW',
                    body={'values': [INDEX_LOG_HEADERS]}
                ).execute()
                return True
                
            except Exception as create_error:
                logger.error(f"Failed to create tab {INDEX_LOG_TAB_NAME}: {create_error}")
                return False
        else:
            logger.error(f"Failed to check/create index headers: {e}")
            return False


def log_index_entry(
    ta_id: str,
    ta_slug: str,
    file_name: str,
    doc_type: str,
    total_pages: int,
    raw_text_length: int,
    chunk_index: int,
    total_chunks: int,
    chunk_text_length: int,
    chunk_context: str,
    chunk_text_preview: str,
    enriched_text_preview: str,
    has_embedding: bool,
    status: str = "success",
    error_message: str = ""
) -> bool:
    """Log a single chunk's indexing details to Google Sheets index_logs tab."""
    if not Config.QA_LOG_SHEET_ID:
        logger.debug("Index logging disabled - no sheet ID configured")
        return False
    
    def _do_log():
        try:
            service = _get_sheets_service()
            if not service:
                logger.warning("Could not get Google Sheets service for index logging")
                return
            
            if not _ensure_index_headers_exist(service, Config.QA_LOG_SHEET_ID):
                return
            
            row = [
                datetime.utcnow().isoformat() + 'Z',
                str(ta_id),
                ta_slug,
                file_name,
                doc_type or "",
                str(total_pages) if total_pages else "",
                str(raw_text_length),
                str(chunk_index),
                str(total_chunks),
                str(chunk_text_length),
                chunk_context[:200] if chunk_context else "",
                chunk_text_preview[:300] if chunk_text_preview else "",
                enriched_text_preview[:300] if enriched_text_preview else "",
                "yes" if has_embedding else "no",
                status,
                error_message[:500] if error_message else ""
            ]
            
            service.spreadsheets().values().append(
                spreadsheetId=Config.QA_LOG_SHEET_ID,
                range=f'{INDEX_LOG_TAB_NAME}!A:A',
                valueInputOption='RAW',
                insertDataOption='INSERT_ROWS',
                body={'values': [row]}
            ).execute()
            
            logger.debug(f"Logged index entry for {file_name} chunk {chunk_index}")
            
        except Exception as e:
            logger.error(f"Failed to log index entry: {e}")
    
    thread = threading.Thread(target=_do_log, daemon=True)
    thread.start()
    return True


def log_index_batch(entries: list) -> bool:
    """Log multiple chunk indexing entries at once (more efficient for large documents)."""
    if not Config.QA_LOG_SHEET_ID:
        logger.debug("Index logging disabled - no sheet ID configured")
        return False
    
    if not entries:
        return True
    
    def _do_log():
        try:
            service = _get_sheets_service()
            if not service:
                logger.warning("Could not get Google Sheets service for index logging")
                return
            
            if not _ensure_index_headers_exist(service, Config.QA_LOG_SHEET_ID):
                return
            
            rows = []
            for entry in entries:
                row = [
                    datetime.utcnow().isoformat() + 'Z',
                    str(entry.get("ta_id", "")),
                    entry.get("ta_slug", ""),
                    entry.get("file_name", ""),
                    entry.get("doc_type", ""),
                    str(entry.get("total_pages", "")) if entry.get("total_pages") else "",
                    str(entry.get("raw_text_length", "")),
                    str(entry.get("chunk_index", "")),
                    str(entry.get("total_chunks", "")),
                    str(entry.get("chunk_text_length", "")),
                    (entry.get("chunk_context", "") or "")[:200],
                    (entry.get("chunk_text_preview", "") or "")[:300],
                    (entry.get("enriched_text_preview", "") or "")[:300],
                    "yes" if entry.get("has_embedding") else "no",
                    entry.get("status", "success"),
                    (entry.get("error_message", "") or "")[:500],
                    (entry.get("headers_found", "") or "")[:500]  # New diagnostic column
                ]
                rows.append(row)
            
            service.spreadsheets().values().append(
                spreadsheetId=Config.QA_LOG_SHEET_ID,
                range=f'{INDEX_LOG_TAB_NAME}!A:A',
                valueInputOption='RAW',
                insertDataOption='INSERT_ROWS',
                body={'values': rows}
            ).execute()
            
            logger.info(f"Logged {len(rows)} index entries to Google Sheets")
            
        except Exception as e:
            logger.error(f"Failed to log index batch: {e}")
    
    thread = threading.Thread(target=_do_log, daemon=True)
    thread.start()
    return True
