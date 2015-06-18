package classificacao;

import java.io.FileReader;
import java.util.Random;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.core.Instance;
import weka.core.Instances;

public class testevinho {
	
	public static void main(String[] args) throws Exception {
				
				FileReader leitor = new FileReader("harmonizacao-vinhos.arff");

				Instances harmonizacao = new Instances(leitor);
				
				harmonizacao.setClassIndex(harmonizacao.numAttributes()-1);
				
				harmonizacao = harmonizacao.resample(new Random());	
				
				Instances baseTreino = harmonizacao.trainCV(3, 0);
				Instances baseTeste = harmonizacao.testCV(2,0);
				Id3 arvore = new Id3();
				NaiveBayes nave = new NaiveBayes();
				
				arvore.buildClassifier(baseTreino);
				nave.buildClassifier(baseTreino);
				
				System.out.println(arvore);
				//System.out.println(nave);

				System.out.println("REAL\tARVORE\tNAIVE BAYES");
				
				for (int e = 0; e < baseTeste.numInstances(); e++) {
					Instance exemplo = baseTeste.instance(e);
					System.out.print(exemplo.classValue());
					exemplo.setClassMissing();
					double classe = arvore.classifyInstance(exemplo);
					System.out.print("\t" + classe);
					classe = nave.classifyInstance(exemplo);
					System.out.println("\t" + classe); 
				}

			}	
	}
