"""Diagnostic tiny fixtures only. None supplies a formal ESCI result."""
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from shoplens.search_benchmark import (GAINS, QRELS_SCRIPT_GAINS, MODELS, canonical_hash,
    declared_protocol, evaluate_scores, make_queries, query_metrics, ranked_indices,
    select_query_ids, tfidf_scores, file_sha, load_search_results, _load_search_results_cached,
    SOURCE_FILES, SOURCE_COMMIT, MODEL_ID, REVISION)


class SearchBenchmarkTests(unittest.TestCase):
    def test_diagnostic_linear_gain_ndcg_matches_hand_calculation(self):
        labels=['I','S','C','E']
        actual=.1/math.log2(3)+.01/math.log2(4)+1/math.log2(5)
        ideal=1+.1/math.log2(3)+.01/math.log2(4)
        result=query_metrics(labels,[4,3,2,1],['a','b','c','d'])
        self.assertAlmostEqual(result['ndcg@10'],actual/ideal)
        self.assertAlmostEqual(result['mrr_exact@10'],.25)
        self.assertEqual(result['recall_exact@10'],1)
        self.assertNotEqual(result['ndcg@10'],result['ndcg@10_qrels_script_variant'])

    def test_diagnostic_ties_use_product_id_not_labels(self):
        self.assertEqual(ranked_indices([1,1,0],['b','a','c']),[1,0,2])
        self.assertEqual(query_metrics(['I','I'],[0,0],['b','a'])['ndcg@10'],0)
        self.assertEqual(query_metrics(['S','C'],[0,0],['b','a'])['mrr_exact@10'],0)
        self.assertEqual(query_metrics(['S','C'],[0,0],['b','a'])['recall_exact@10'],0)

    def test_diagnostic_exact_recall_cutoff_is_not_hit_rate(self):
        labels=['E','I','E','E']
        r=query_metrics(labels,[4,3,2,1],list('abcd'),k=2)
        self.assertEqual(r['recall_exact@10'],1/3)
        self.assertEqual(r['mrr_exact@10'],1)

    def test_diagnostic_bad_scores_labels_and_duplicate_ids_fail(self):
        for scores in [[np.nan,0],[np.inf,0],[1]]:
            with self.subTest(scores=scores),self.assertRaises(ValueError):
                query_metrics(['E','I'],scores,['a','b'])
        with self.assertRaises(ValueError): query_metrics(['E','?'],[1,0],['a','b'])
        with self.assertRaises(ValueError): ranked_indices([1,2],['a','a'])

    def test_diagnostic_selection_is_label_text_and_row_order_blind(self):
        rows=[{'query_id':i,'product_id':p,'split':s,'esci_label':'E','query':'q'}
              for s in ['train','test'] for i in range(20) for p in ['a','b']]
        ids=select_query_ids(rows,'test',5)
        changed=[dict(r,esci_label='I',query='different') for r in reversed(rows)]
        self.assertEqual(ids,select_query_ids(changed,'test',5))
        self.assertNotEqual(ids,select_query_ids(rows,'train',5))
        with self.assertRaises(ValueError): select_query_ids(rows,'test',21)

    def test_diagnostic_complete_pool_and_join_validation(self):
        rows=[{'query_id':1,'query':'red shoe','product_id':p,'esci_label':l}
              for p,l in [('b','I'),('a','E')]]
        q=make_queries(rows,['1'],{'a':'Red shoe','b':'Blue shoe'})[0]
        self.assertEqual(q['official_candidate_count'],2)
        self.assertEqual([x['product_id'] for x in q['candidates']],['a','b'])
        with self.assertRaises(ValueError): make_queries(rows,['1'],{'a':'Red shoe'})
        with self.assertRaises(ValueError): make_queries(rows+rows[:1],['1'],{'a':'A','b':'B'})

    def test_diagnostic_tfidf_fits_training_titles_only(self):
        train=[{'query':'unimportant','candidates':[{'product_id':'a','product_title':'red shoes'},
                                                  {'product_id':'b','product_title':'blue coat'}]}]
        test=[{'query':'nevertrainword','candidates':[{'product_id':'x','product_title':'nevertrainword'},
                                                    {'product_id':'y','product_title':'red shoes'}]}]
        self.assertTrue(np.array_equal(tfidf_scores(train,test)[0],np.zeros(2)))
        test[0]['query']='red shoes'
        scores=tfidf_scores(train,test)[0]
        self.assertEqual(scores[0],0)
        self.assertGreater(scores[1],.99)

    def test_diagnostic_all_four_models_required_no_fallback(self):
        queries=[{'query_id':'1','query':'shoe','candidates':[
            {'product_id':'a','product_title':'Shoe','esci_label':'E','gain':1},
            {'product_id':'b','product_title':'Coat','esci_label':'I','gain':0}]}]
        scores={m:[[1.,0.]] for m in MODELS}
        rows,models=evaluate_scores(queries,scores)
        self.assertEqual(len(models),4)
        self.assertTrue(all(m['metrics']['ndcg@10']==1 for m in models))
        self.assertTrue(all(m['paired_ndcg10_vs_frozen']['ties']==1 for m in models))
        self.assertEqual(rows[0]['candidates'][0]['ranks']['qwen_preference'],1)
        self.assertNotIn('scores',queries[0]['candidates'][0])
        scores.pop('qwen_preference')
        with self.assertRaises(ValueError): evaluate_scores(queries,scores)

    def test_declared_protocol_separates_mapping_conflict_and_transfer(self):
        p=declared_protocol()
        self.assertEqual(p['test_queries'],500)
        self.assertEqual(p['gains'],{'E':1,'S':.1,'C':.01,'I':0})
        self.assertEqual(p['sensitivity_gains'],QRELS_SCRIPT_GAINS)
        self.assertEqual(p['adapter_domain'],'clothing')
        self.assertFalse(p['images_used'])
        self.assertEqual(canonical_hash(p),canonical_hash(declared_protocol()))


class RangeReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        location=Path(__file__).parents[1]/'scripts/run_esci_benchmark.py'
        spec=importlib.util.spec_from_file_location('diagnostic_esci_runner',location)
        cls.runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.runner)

    def response(self,status=206,range_value='bytes 2-4/10'):
        class Response:
            status_code=status
            headers={'Content-Range':range_value}
            raw=__import__('io').BytesIO(b'cde')
            def __enter__(self):return self
            def __exit__(self,*args):pass
        return Response()

    def test_diagnostic_exact_range_cache_and_tamper_refetch(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.runner.RangeFile('https://example.invalid/public.parquet',10,d)
            with patch.object(r.session,'get',side_effect=lambda *a,**k:self.response()) as get:
                r.seek(2);self.assertEqual(r.read(3),b'cde');self.assertEqual(r.network_bytes,3)
                r.seek(2);self.assertEqual(r.read(3),b'cde');self.assertEqual(get.call_count,1)
                (Path(d)/'2-4.bin').write_bytes(b'xxx')
                r.seek(2);self.assertEqual(r.read(3),b'cde');self.assertEqual(get.call_count,2)

    def test_diagnostic_refuses_full_file_response_and_budget_overrun(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.runner.RangeFile('https://example.invalid/public.parquet',10,d,budget=3)
            r.seek(2)
            with patch.object(r.session,'get',return_value=self.response(status=200)),patch.object(self.runner.time,'sleep'):
                with self.assertRaises(RuntimeError):r.read(3)
            r.seek(0)
            with patch.object(r.session,'get') as get,self.assertRaises(RuntimeError):r.read(4)
            get.assert_not_called()

    def test_diagnostic_real_unicode_separator_jsonl_round_trip(self):
        import json
        row={'product_title':'Title\u2028contains\u0085Unicode separators'}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'fixture.jsonl'
            self.runner.write_jsonl(path,[row,row])
            lines=path.read_text().splitlines()
            self.assertEqual(len(lines),2)
            self.assertEqual([json.loads(x) for x in lines],[row,row])


class ResultLoaderTests(unittest.TestCase):
    """500 repetitions of a two-candidate DIAGNOSTIC fixture, in temp dirs only."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.out=self.root/'results/esci';self.out.mkdir(parents=True)
        p=declared_protocol(); h=canonical_hash(p)
        self.inputs=[{'query_id':str(i),'query':'diagnostic fixture','locale':'us','official_candidate_count':2,
            'candidates':[{'product_id':'a','product_title':'Diagnostic shoe','esci_label':'E','gain':1.0},
                          {'product_id':'b','product_title':'Diagnostic coat','esci_label':'I','gain':0.0}]} for i in range(500)]
        train=[dict(q,query_id='train-'+q['query_id'],candidates=[{k:v for k,v in c.items() if k not in ('esci_label','gain')} for c in q['candidates']]) for q in self.inputs]
        self.write_lines('test_inputs.jsonl',self.inputs)
        self.write_lines('tfidf_training_inputs.jsonl',train)
        self.rows,self.models=evaluate_scores(self.inputs,{m:[[1.,0.] if m!='qwen_semantic' else [0.,1.] for q in self.inputs] for m in MODELS})
        self.write_lines('queries.jsonl',self.rows)
        ids=[q['query_id'] for q in self.inputs]
        exports={m:{'model_id':MODEL_ID,'revision':REVISION,'domain':'clothing','seed':42,'selected_epoch':2,
                    'head_used':False,'adapter_sha256':None if m=='qwen_frozen' else 'a'*64} for m in MODELS[1:]}
        source={'commit':SOURCE_COMMIT,'examples':dict(SOURCE_FILES['examples'],full_file_sha256_verified=True),
                'products':SOURCE_FILES['products'],'test_inputs_sha256':file_sha(self.out/'test_inputs.jsonl'),
                'tfidf_training_inputs_sha256':file_sha(self.out/'tfidf_training_inputs.jsonl'),'query_ids_sha256':canonical_hash(ids)}
        self.summary={'schema_version':1,'benchmark_id':p['benchmark_id'],'complete':True,'diagnostic':False,
            'protocol':p,'protocol_sha256':h,'source_provenance':source,'models':self.models,'model_exports':exports,
            'query_count':500,'candidate_count':1000,'unique_test_products':2,'query_ids_sha256':canonical_hash(ids),
            'queries_sha256':file_sha(self.out/'queries.jsonl')}
        self.write('summary.json',self.summary)
        self.write('source_provenance.json',source)
        self.write('protocol.json',{'protocol':p,'protocol_sha256':h,'model_exports':exports})
        self.write('selected_query_ids.json',{'test':ids,'train':[q['query_id'] for q in train],
                    'protocol_sha256':h,'selection_uses_labels':False})

    def tearDown(self):
        _load_search_results_cached.cache_clear();self.temp.cleanup()

    def write(self,name,value):
        (self.out/name).write_text(json.dumps(value,allow_nan=False))

    def write_lines(self,name,rows):
        (self.out/name).write_text(''.join(json.dumps(r,ensure_ascii=True,allow_nan=False)+'\n' for r in rows))

    def changed_rows(self):
        self.write_lines('queries.jsonl',self.rows)
        self.summary['queries_sha256']=file_sha(self.out/'queries.jsonl')
        self.write('summary.json',self.summary)

    def test_diagnostic_full_replay_no_torch_and_cache_defensive_copy(self):
        import builtins
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name=='torch' or name.startswith('transformers'):
                raise AssertionError('Result loader must not import a model runtime')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded):
            first=load_search_results(self.root)
            hits=_load_search_results_cached.cache_info().hits
            second=load_search_results(self.root)
        self.assertEqual(_load_search_results_cached.cache_info().hits,hits+1)
        first['queries'][0]['candidates'][0]['product_title']='mutated UI state'
        self.assertEqual(second['queries'][0]['candidates'][0]['product_title'],'Diagnostic shoe')
        self.assertEqual(len(second['queries']),500)
        self.assertEqual(second['signature'],load_search_results(self.root)['signature'])

    def test_diagnostic_summary_mutation_invalidates_cache_and_fails_replay(self):
        load_search_results(self.root)
        self.summary['models'][0]['metrics']['ndcg@10']=.123
        self.write('summary.json',self.summary)
        with self.assertRaisesRegex(ValueError,'recomputed evidence'):load_search_results(self.root)

    def test_diagnostic_score_hash_and_original_label_tampering_rejected(self):
        with (self.out/'queries.jsonl').open('a') as f:f.write('{}\n')
        with self.assertRaisesRegex(ValueError,'SHA256'):load_search_results(self.root)
        self.rows[0]['candidates'][0]['esci_label']='I';self.changed_rows()
        with self.assertRaisesRegex(ValueError,'labels or product text'):load_search_results(self.root)

    def test_diagnostic_rehashed_rank_and_query_metric_tampering_rejected(self):
        self.rows[0]['candidates'][0]['ranks']['qwen_frozen']=2;self.changed_rows()
        with self.assertRaisesRegex(ValueError,'permutation'):load_search_results(self.root)
        self.rows[0]['candidates'][0]['ranks']['qwen_frozen']=1
        self.rows[0]['metrics']['qwen_frozen']['ndcg@10']=.2;self.changed_rows()
        with self.assertRaisesRegex(ValueError,'recomputed evidence'):load_search_results(self.root)

    def test_diagnostic_incomplete_wrong_model_and_nonfinite_scores_rejected(self):
        self.summary['complete']=False;self.write('summary.json',self.summary)
        with self.assertRaisesRegex(ValueError,'incomplete'):load_search_results(self.root)
        self.summary['complete']=True;self.summary['models'][0]['model']='other';self.write('summary.json',self.summary)
        with self.assertRaisesRegex(ValueError,'four declared'):load_search_results(self.root)
        self.summary['models'][0]['model']='tfidf'
        self.rows[0]['candidates'][0]['scores']['tfidf']=float('nan')
        (self.out/'queries.jsonl').write_text(''.join(json.dumps(r,allow_nan=True)+'\n' for r in self.rows))
        self.summary['queries_sha256']=file_sha(self.out/'queries.jsonl');self.write('summary.json',self.summary)
        with self.assertRaisesRegex(ValueError,'finite numeric'):load_search_results(self.root)


if __name__=='__main__':unittest.main()
