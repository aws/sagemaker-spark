/*
 * Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 *   http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */

package com.amazonaws.services.sagemaker.sparksdk.protobuf;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.io.OutputStream;

/**
 * A Hadoop {@link FileOutputFormat} that writes {@link BytesWritable} as Amazon Record protobuf
 * messages in a recordIO encoded file.
 *
 * @see [[https://aws.amazon.com/sagemaker/latest/dg/cdf-training.html/]] for more information on
 *     the Amazon Record data format.
 * @see [[https://mxnet.incubator.apache.org/architecture/note_data_loading.html]] for more
 *     information on recordIO
 */
public class RecordIOOutputFormat extends FileOutputFormat<NullWritable, BytesWritable> {

    public static class SageMakerProtobufRecordWriter extends
            RecordWriter<NullWritable, BytesWritable> {

        private OutputStream out;

        public SageMakerProtobufRecordWriter(OutputStream out) {
            this.out = out;
        }

        @Override
        public void write(NullWritable nullWritable, BytesWritable bytesWritable)
                throws IOException, InterruptedException {

            byte[] bytes = ProtobufConverter.byteArrayToRecordIOEncodedByteArray(bytesWritable.getBytes());
            out.write(bytes, 0, bytes.length);
        }


        @Override
        public void close(TaskAttemptContext taskAttemptContext) throws IOException, InterruptedException {
            this.out.close();
        }
    }

    @Override
    public RecordWriter<NullWritable, BytesWritable> getRecordWriter(TaskAttemptContext taskAttemptContext)
            throws IOException, InterruptedException {
        Path file = getDefaultWorkFile(taskAttemptContext, "sagemaker");
        FileSystem fs = file.getFileSystem(taskAttemptContext.getConfiguration());

        FSDataOutputStream out = fs.create(file, true);
        return new SageMakerProtobufRecordWriter(out);
    }
}
