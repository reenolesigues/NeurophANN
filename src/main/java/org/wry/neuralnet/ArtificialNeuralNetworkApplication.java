package org.wry.neuralnet;

/*
 * Reference: https://www.baeldung.com/neuroph
 */

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.wry.neuralnet.structure.NeuralNetworkOrchestrator;
import org.wry.neuralnet.structure.NeuralNetworkDemo;

@SpringBootApplication
public class ArtificialNeuralNetworkApplication {

	public static void main(String[] args) throws Exception {
//		SpringApplication.run(ArtificialNeuralNetworkApplication.class, args);
		
		NeuralNetworkOrchestrator orchestrator = new NeuralNetworkOrchestrator(2,4,4,1);
		orchestrator.forwardPropagate(0, 0);
		orchestrator.backPropagate();
		orchestrator.forwardPropagate(0, 0);
	}
}