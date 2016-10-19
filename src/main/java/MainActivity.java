import java.io.File;

import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.columns.ColumnDefinition;
import org.encog.ml.data.versatile.columns.ColumnType;
import org.encog.ml.data.versatile.sources.CSVDataSource;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.model.EncogModel;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

public class MainActivity {
	private static EncogModel model;

	private static BasicNetwork createNetwork() {
		boolean isBias = true;

		BasicLayer[] layers = { new BasicLayer(new ActivationTANH(), isBias, 2),
				new BasicLayer(new ActivationTANH(), isBias, 50), new BasicLayer(new ActivationTANH(), isBias, 3) };

		BasicNetwork network = new BasicNetwork();
		for (BasicLayer l : layers)
			network.addLayer(l);
		network.getStructure().finalizeStructure();
		network.reset();

		return network;
	}

	private static VersatileMLDataSet getDataSet(String filename) {
		File file = new File(filename);
		VersatileMLDataSet set = new VersatileMLDataSet(new CSVDataSource(file, true, CSVFormat.DECIMAL_POINT));
		set.getNormHelper().setFormat(CSVFormat.DECIMAL_POINT);

		set.defineSourceColumn("x", 0, ColumnType.continuous);
		set.defineSourceColumn("y", 1, ColumnType.continuous);
		ColumnDefinition cls = set.defineSourceColumn("cls", 2, ColumnType.nominal);
		cls.defineClass(new String[] { "1", "2", "3" });
		set.analyze();
		set.defineSingleOutputOthersInput(cls);

		model = new EncogModel(set);
		model.selectMethod(set, MLMethodFactory.TYPE_FEEDFORWARD);
		set.normalize();
		model.holdBackValidation(0.3, true, 1001);
		model.selectTrainingType(set);

		return set;
	}

	private static BasicNeuralDataSet getNeuralDataSet(VersatileMLDataSet set) {
		double[][] inputArray = new double[model.getDataset().getData().length][];
		double[][] outputArray = new double[model.getDataset().getData().length][];
		int j = 0;
		for (MLDataPair mlDataPair : model.getDataset()) {
			inputArray[j] = mlDataPair.getInputArray();
			outputArray[j] = mlDataPair.getIdealArray();
			j++;
		}

		return new BasicNeuralDataSet(inputArray, outputArray);
	}

	public static void main(String[] args) {
		int iterationsCount = 2000;
		double learningRate = 0.05, momentum = 0.01, targetError = 0.025;
		String trainingDataSetFilename = "/tmp/data.train.csv";
		VersatileMLDataSet set = getDataSet(trainingDataSetFilename);

		System.out.println("Starting classification!");
		NeuralNetwork network = new NeuralNetwork(learningRate, momentum, createNetwork(), getNeuralDataSet(set));
		network.trainNetwork(iterationsCount, targetError);
		network.getClassificationResults(new ReadCSV(trainingDataSetFilename, true, CSVFormat.DECIMAL_POINT), set);
	}

}
