import unittest

from src.models.predict_model import prever_risco

class TestSmoke(unittest.TestCase):
    
    def test_predict_retorna_dicionario(self):
        """Teste simples para ver se a função retorna o tipo esperado"""
        resultado = prever_risco({}, "modelo_fake")
        
        self.assertIsInstance(resultado, dict)
        self.assertIn("risco", resultado)
        self.assertIn("probabilidade", resultado)

    def test_probabilidade_valida(self):
        """Teste para garantir que a probabilidade está entre 0 e 1"""
        resultado = prever_risco({}, "modelo_fake")
        prob = resultado["probabilidade"]
        
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

if __name__ == '__main__':
    unittest.main()