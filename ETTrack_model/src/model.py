# if framenum == 0:
            #     #temporal_input_embedded_relu = self.relu(self.input_embedding_layer_temporal(node_abs))
            #     tcn_input = node_abs.transpose(1, 2)
            #     temporal_input_embedded_relu = self.relu(self.tcn(tcn_input))
            #     temporal_input_embedded_relu = temporal_input_embedded_relu.transpose(1, 2)
            #     temporal_input_embedded = self.dropout_in(temporal_input_embedded_relu.clone()) # (1,323,32)
            # else:
            #     #temporal_input_embedded_relu = self.relu(self.input_embedding_layer_temporal(node_abs))
            #     tcn_input = node_abs.transpose(1, 2)
            #     temporal_input_embedded_relu = self.relu(self.tcn(tcn_input))
            #     temporal_input_embedded_relu = temporal_input_embedded_relu.transpose(1, 2)
            #     temporal_input_embedded = self.dropout_in(temporal_input_embedded_relu.clone())
            #
            # temporal_input_embedded = self.temporal_encoder_1(temporal_input_embedded)
            # temporal_input_embedded_last = temporal_input_embedded[-1] #(323,32)