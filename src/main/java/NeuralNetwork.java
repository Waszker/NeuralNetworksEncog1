import java.io.FileWriter;
import java.io.IOException;

import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.util.csv.ReadCSV;

public class NeuralNetwork {
	private boolean isTrained = false;
	BasicNetwork network;
	private Backpropagation trainer;

	public NeuralNetwork(double learningRate, double momentum, BasicNetwork network, BasicNeuralDataSet trainingSet) {
		super();
		this.network = network;
		this.trainer = new Backpropagation(network, trainingSet, learningRate, momentum);
	}

	public void trainNetwork(int iterationsCount, double targetError) {
		// TODO: Implement array for training errors
		if (trainer != null) {
			trainer.setBatchSize(1);
			while (true) {
				isTrained = true;
				trainer.iteration();
				if (trainer.getIteration() % 100 == 0)
					System.out.println("Iteration: " + trainer.getIteration() + " Error -> " + trainer.getError());
				if (trainer.getError() < targetError || --iterationsCount <= 0)
					break;
			}
			System.out.println("Finished training with error: " + trainer.getError());
			trainer.finishTraining();
		}
	}

	public void getClassificationResults(ReadCSV csvReader, VersatileMLDataSet dataSet, String resultsFilename)
			throws IOException {

		String csvFile = resultsFilename;
		FileWriter writer = new FileWriter(csvFile);
		StringBuilder sb = new StringBuilder();
		String columns = "x,y,cls\n";
		sb.append(columns);

		NormalizationHelper norm = dataSet.getNormHelper();
		int correct = 0, all = 0, inputVectorLength = csvReader.getColumnNames().size() - 1;
		String[] line = new String[inputVectorLength + 1];
		double[] data = new double[inputVectorLength + 1];
		MLData output;

		while (isTrained && csvReader.next()) {
			for (int i = 0; i < line.length; i++) {
				String value = csvReader.get(i);
				line[i] = value;
				data[i] = Double.parseDouble(value);

				sb.append(value);
				sb.append(",");
			}
			norm.normalizeInputVector(line, data, true);
			output = network.compute(new BasicMLData(data));

			double predictedValue = Double.parseDouble(norm.denormalizeOutputVectorToString(output)[0]);
			double predictedClass = Math.round(predictedValue);
			double correctClass = Double.parseDouble(csvReader.get(inputVectorLength));
			// System.out.println("Got " + predictedClass + " for " +
			// correctClass);
			if (predictedClass == correctClass)
				correct++;
			all++;

			sb.append(String.valueOf(predictedClass));
			sb.append("\n");
		}

		System.out.println("Final score: " + (correct * 100.0) / all + "%");
		writer.append(sb.toString());
		writer.flush();
		writer.close();
	}

	public void getRegressionResults(ReadCSV csvReader, VersatileMLDataSet dataSet, String resultsFilename)
			throws IOException {

		String csvFile = resultsFilename;
		FileWriter writer = new FileWriter(csvFile);
		StringBuilder sb = new StringBuilder();
		String columns = "x,y\n";
		sb.append(columns);

		NormalizationHelper norm = dataSet.getNormHelper();
		int inputVectorLength = csvReader.getColumnNames().size() - 1;
		String[] line = new String[inputVectorLength + 1];
		double[] data = new double[inputVectorLength + 1];
		MLData output;

		while (isTrained && csvReader.next()) {
			for (int i = 0; i < line.length; i++) {
				String value = csvReader.get(i);
				line[i] = value;
				data[i] = Double.parseDouble(value);

				sb.append(value);
				sb.append(",");
			}
			norm.normalizeInputVector(line, data, true);
			output = network.compute(new BasicMLData(data));

			double predictedValue = Double.parseDouble(norm.denormalizeOutputVectorToString(output)[0]);

			sb.append(String.valueOf(predictedValue));
			sb.append("\n");
		}
		System.out.println("REGRESSION DONE");
		writer.append(sb.toString());
		writer.flush();
		writer.close();
	}

}
