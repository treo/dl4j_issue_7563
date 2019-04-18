package tech.dubs;

import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.SelfAttentionLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class App
{
    public static void main( String[] args )
    {
        int nLayers = 2;
        double l2 = 1e-5;
        int in = 80;
        int out = 80;
        double dropout = 1.0;
        int tbpttLength = 35;
        int layerSize = 256;
        double lr = 1e-3;

        String[] outputLayerInputs = new String[nLayers];
        ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .l2(l2)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(lr))
                .graphBuilder()
                .addInputs("input")
                .addLayer("layer-0", new LSTM.Builder()
                        .nIn(in)
                        .nOut(layerSize)
                        .activation(Activation.TANH)
                        .dropOut(dropout)
                        .build(), "input");

        outputLayerInputs[0] = "layer-0";
        for (int i = 1; i < nLayers; i++) {
            builder.addLayer("layer-" + i, new LSTM.Builder()
                    .nIn(layerSize)
                    .nOut(layerSize)
                    .dropOut(dropout)
                    .activation(Activation.TANH)
                    .build(), "layer-" + (i - 1));
            outputLayerInputs[i] = "layer-" + i;
        }
        builder.addLayer("attention", new SelfAttentionLayer.Builder()
                .nHeads(1)
                .projectInput(true)
                .nIn(nLayers * layerSize)
                .nOut(layerSize)
                .build(), outputLayerInputs);
        ComputationGraphConfiguration conf = builder
                .addLayer("outputLayer", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(layerSize)
                        .nOut(out)
                        .build(), "attention")
                .setOutputs("outputLayer")
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength)
                .tBPTTBackwardLength(tbpttLength)
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        final DataSet data = new DataSet(Nd4j.rand('f', new int[]{128, 80, 1050}), Nd4j.rand('f', new int[]{128, 80, 1050}));
        net.fit(data);
    }
}
