
package classificacao;

import java.io.FileReader;
import java.util.Random;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class Classificacao {

    public static void main(String[] args) throws Exception{
        
		// Lendo os exemplos a partir do arquivo estado-nutricional.arff
		FileReader leitor = new FileReader("estado-nutricional.arff");
		Instances base = new Instances(leitor);
		
		// Definindo o indice do atributo classe (ultimo atributo do conjunto)
		base.setClassIndex(base.numAttributes() - 1);
		
		// Criando uma nova base com os exemplos embaralhados 
		base = base.resample(new Random());			
		
		// Abordagem Hold out de validaÃ§Ã£o cruzada 
		Instances baseTeste = base.testCV(3, 0); // Obtendo subconjunto para testes
		Instances baseTreino = base.trainCV(3, 0); // Obtendo subconjunto para treinamento
		
		// Criando os classificadores que serÃ£o avaliados
		IBk knn = new IBk(5); // knn com 3 vizinhos
		IB1 vizinho = new IB1(); // vizinho mais prÃ³ximo
		
		// Treinando os classificadores instanciados
		knn.buildClassifier(baseTreino);
		vizinho.buildClassifier(baseTreino);
		
		System.out.println("real\tknn\tvizinho"); // imprimindo rotulos para as colunas
		
		for (int e = 0; e < baseTeste.numInstances(); e++) {
			Instance exemplo = baseTeste.instance(e);
			System.out.print(exemplo.classValue()); // imprimindo o valor da classe real do exemplo
			exemplo.setClassMissing(); // removendo informacao da classe
			double classe = knn.classifyInstance(exemplo); // resposta do knn
			System.out.print("\t" + classe); // imprimindo resposta do knn
			classe = vizinho.classifyInstance(exemplo); // resposta do vizinho mais prÃ³ximo
			System.out.println("\t" + classe); // imprimindo resposta do vizinho mais prÃ³ximo
		}
	}
}


