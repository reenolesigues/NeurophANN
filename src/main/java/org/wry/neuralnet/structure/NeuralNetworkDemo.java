package org.wry.neuralnet.structure;

/**
 * DEMO ONLY
 */

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.ConnectionFactory;

public class NeuralNetworkDemo {

	public NeuralNetworkDemo() {
	
		/*Input Layer Creation*/
		Layer inputLayer = new Layer();
		inputLayer.addNeuron(new Neuron());
		inputLayer.addNeuron(new Neuron());
		
		/*Hidden Layer 1 Creation*/
		Layer hiddenLayerOne = new Layer();
		hiddenLayerOne.addNeuron(new Neuron());
		hiddenLayerOne.addNeuron(new Neuron());
		hiddenLayerOne.addNeuron(new Neuron());
		hiddenLayerOne.addNeuron(new Neuron());
		
		/*Hidden Layer 2 Creation*/
		Layer hiddenLayerTwo = new Layer();
		hiddenLayerTwo.addNeuron(new Neuron());
		hiddenLayerTwo.addNeuron(new Neuron());
		hiddenLayerTwo.addNeuron(new Neuron());
		hiddenLayerTwo.addNeuron(new Neuron());
		
		/*Output Layer Creation*/
		Layer outputLayer = new Layer();
		outputLayer.addNeuron(new Neuron());
		
		/*Structuring the Artificial Neural Network*/
		NeuralNetwork<LearningRule> ann = new NeuralNetwork<LearningRule>();
		ann.addLayer(0, inputLayer);
		ann.addLayer(1, hiddenLayerOne);
		/*Weave simultaneous layers*/
		ConnectionFactory.fullConnect(ann.getLayerAt(0), ann.getLayerAt(1));
		ann.addLayer(2, hiddenLayerTwo);
		ConnectionFactory.fullConnect(ann.getLayerAt(1), ann.getLayerAt(2));
		ann.addLayer(3, outputLayer);
		ConnectionFactory.fullConnect(ann.getLayerAt(2), ann.getLayerAt(3));
		/*Weave the input layer to the output layer to complete/close the neural network */
		ConnectionFactory.fullConnect(ann.getLayerAt(0), ann.getLayerAt(ann.getLayersCount()-1), false);
		ann.setInputNeurons(inputLayer.getNeurons());
		ann.setOutputNeurons(outputLayer.getNeurons());
		
		/*Training Part*/
		int inputSize = inputLayer.getNeuronsCount();
		int outputSize = outputLayer.getNeuronsCount();
		DataSet ds = new DataSet(inputSize, outputSize);
		
		DataSetRow rOne = new DataSetRow(new double[] {0, 0}, new double[] {0});
		DataSetRow rTwo = new DataSetRow(new double[] {0, 1}, new double[] {0});
		DataSetRow rThree = new DataSetRow(new double[] {1, 0}, new double[] {0});
		DataSetRow rFour = new DataSetRow(new double[] {1, 1}, new double[] {1});
		
		ds.addRow(rOne);
		ds.addRow(rTwo);
		ds.addRow(rThree);
		ds.addRow(rFour);
		
		BackPropagation backPropagation = new BackPropagation();
		backPropagation.setMaxIterations(1000);
		ann.learn(ds, backPropagation);
		
		/*Run FFANN on sample vector inputs*/
		ann.setInput(0, 0);
		ann.calculate();
		double[] networkOutputOne = ann.getOutput();
		for (double d : networkOutputOne) {
			System.out.println(d);
		}
	}
	
}
