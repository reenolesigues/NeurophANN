package org.wry.neuralnet.structure;

import java.util.stream.IntStream;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.ConnectionFactory;

public class NeuralNetworkOrchestrator {

	private NeuralNetwork<LearningRule> ann;
	
	public NeuralNetworkOrchestrator(int... neuronsPerLayerSequence) throws Exception {
		analyzeDesiredNNModel(neuronsPerLayerSequence);
		configureNeuralNetwork(neuronsPerLayerSequence);
	}

	private void analyzeDesiredNNModel(int[] neuronsPerLayerSequence) throws Exception {
		if(neuronsPerLayerSequence.length < 3) {
			throw new Exception("Layers should be at least 3 (input, hidden, output)");
		}
		for(int i = 1; i<neuronsPerLayerSequence.length - 1; i++) {
			if(neuronsPerLayerSequence[i] > (neuronsPerLayerSequence[1] * 2) ) {
				throw new Exception("Neurons per layer should not exceed twice the number pf input layer neurons ideally");
			}
			if(neuronsPerLayerSequence[i] < 2 && i != neuronsPerLayerSequence.length-1) {
				throw new Exception("Hidden layers should have at least 2 neurons");
			}
		}
	}
	
	private void configureNeuralNetwork(int[] neuronsPerLayerSequence) {
		ann = new NeuralNetwork<LearningRule>();
		for(int i = 0; i < neuronsPerLayerSequence.length; i++) {
			Layer layer = new Layer();
			IntStream.range(0,neuronsPerLayerSequence[i]).forEach(x -> layer.addNeuron(new Neuron()));
			ann.addLayer(i, layer);
			/*Configure all other layers up tp output excluding input*/
			if(i > 0) {
				ConnectionFactory.fullConnect(ann.getLayerAt(i-1), ann.getLayerAt(i), false);
			}
			/*Configure output layer*/
			if(i == neuronsPerLayerSequence.length - 1) {
				ConnectionFactory.fullConnect(ann.getLayerAt(0), ann.getLayerAt(i));
			}
		}
		ann.setInputNeurons(ann.getLayerAt(0).getNeurons());
		ann.setOutputNeurons(ann.getLayerAt(ann.getLayersCount()-1).getNeurons());
	}
	
	/*Training*/
	public void backPropagate() {
		System.out.println(">>> training ...");
		int inputSize = ann.getLayerAt(0).getNeuronsCount();
		int outputSize = ann.getLayerAt(ann.getLayersCount()-1).getNeuronsCount();
		
		/* Can be loaded from file later
		 * Note: Sizes must match the number of input neurons and output neurons
		 */
		DataSet ds = new DataSet(inputSize, outputSize);
		DataSetRow rOne = new DataSetRow(new double[] {1, 0}, new double[] {1});
		DataSetRow rTwo = new DataSetRow(new double[] {0, 0}, new double[] {0});
		DataSetRow rThree = new DataSetRow(new double[] {1, 1}, new double[] {0});
		DataSetRow rFour = new DataSetRow(new double[] {0, 1}, new double[] {1});
		
		ds.addRow(rOne);
		ds.addRow(rTwo);
		ds.addRow(rThree);
		ds.addRow(rFour);
		
		BackPropagation backPropagation = new BackPropagation();
		backPropagation.setMaxIterations(1000);
		ann.learn(ds, backPropagation);
	}
	
	public void forwardPropagate(double... inputs) {
		System.out.println(">> forward propagating ...");
		ann.setInput(inputs);
		ann.calculate();
		double[] outputs = ann.getOutput();
		IntStream.range(0, outputs.length).forEach(x -> System.out.print(outputs[x] + " | "));
		System.out.println();
	}
	
}
