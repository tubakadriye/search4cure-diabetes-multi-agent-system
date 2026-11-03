"""Execute isolated legacy function bodies without importing Google clients."""
import ast
from pathlib import Path
import subprocess


def extract(source, name, namespace):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node],type_ignores=[]),'legacy','exec'),namespace)
    return namespace[name]


def image_summaries(source):
    namespace = dict(Optional=__import__('typing').Optional,db={'docs_multimodal':object()},
                     COLLECTION_NAME='docs_multimodal',vector_search=lambda *a,**kw:[1,2,3],
                     gemini_page_image_summary=lambda doc,collection:str(doc))
    return extract(source,'vector_search_image_tool',namespace)(text_query='test')


def test_summary_bug_reproduced_and_fixed():
    path = 'backend/sub_agents/article_agents.py'
    before = subprocess.check_output(['git','show','8a93bf295d1116563171a1861681fc48848a43fb:'+path],text=True)
    assert image_summaries(before) == '3'
    assert image_summaries(Path(path).read_text()) == '1\n\n2\n\n3'


def test_legacy_projection_keeps_cache_id():
    namespace = dict(Union=__import__('typing').Union,Image=type('Image',(),{'Image':type('Img',(),{})}),
                     List=list,Dict=dict,Any=object,VS_INDEX_NAME='test',get_sbert_embedding=lambda q:[1],
                     get_image_from_gcs=lambda *a:b'',gcs_bucket=object(),_summarize_with_gemini=lambda t:{})
    class Collection:
        def aggregate(self, pipeline):
            assert pipeline[1]['$project']['_id'] == 1
            return []
    fn = extract(Path('backend/retriever/rag_retriever.py').read_text(),'vector_search',namespace)
    assert fn('test','sbert',Collection(),False) == []


def test_main_import_names_exist():
    main = ast.parse(Path('backend/main.py').read_text())
    runner = ast.parse(Path('backend/agent/agent_runner.py').read_text())
    definitions = {n.name for n in runner.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    for node in ast.walk(main):
        if isinstance(node,ast.ImportFrom) and node.module == 'agent.agent_runner':
            assert {n.name for n in node.names} <= definitions
