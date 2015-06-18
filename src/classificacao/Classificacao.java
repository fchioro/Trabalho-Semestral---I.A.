/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classificacao;

import java.io.FileReader;
import java.util.Random;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class Classificacao {

    public static void main(String[] args) throws Exception{
        
		// Lendo os exemplos a partir do arquivo iris.arff
		FileReader leitor = new FileReader("C:\\estado-nutricional.arff");
		Instances iris = new Instances(leitor);
		
		// Definindo o Ã­ndice do atributo classe (Ãºltimo atributo do conjunto)
		iris.setClassIndex(iris.numAttributes() - 1);
		
		// Criando uma nova base com os exemplos embaralhados 
		iris = iris.resample(new Random());			
		
		// Abordagem Hold out de validaÃ§Ã£o cruzada 
		Instances baseTeste = iris.testCV(3, 0); // Obtendo subconjunto para testes
		Instances baseTreino = iris.trainCV(3, 0); // Obtendo subconjunto para treinamento
		
		// Criando os classificadores que serÃ£o avaliados
		IBk knn = new IBk(5); // knn com 3 vizinhos
		IB1 vizinho = new IB1(); // vizinho mais prÃ³ximo
		
		// Treinando os classificadores instanciados
		knn.buildClassifier(baseTreino);
		vizinho.buildClassifier(baseTreino);
		
		System.out.println("real\tknn\tvizinho"); // imprimindo rÃ³tulos para as colunas
		
		for (int e = 0; e < baseTeste.numInstances(); e++) {
			Instance exemplo = baseTeste.instance(e);
			System.out.print(exemplo.classValue()); // imprimindo o valor da classe real do exemplo
			exemplo.setClassMissing(); // removendo informaÃ§Ã£o da classe
			double classe = knn.classifyInstance(exemplo); // resposta do knn
			System.out.print("\t" + classe); // imprimindo resposta do knn
			classe = vizinho.classifyInstance(exemplo); // resposta do vizinho mais prÃ³ximo
			System.out.println("\t" + classe); // imprimindo resposta do vizinho mais prÃ³ximo
		}
	}
}


    }
    
}
