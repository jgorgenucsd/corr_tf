#!/usr/bin/env python
"""
Tests for the correlation Tensorflow operation.

.. moduleauthor:: Justin
"""

import unittest
import numpy as np
import tensorflow as tf
import correlation_layer as cl

class CorrelationOpTest(unittest.TestCase):
    def test_raisesExceptionWithIncompatibleDimensions(self):
        ''' correlation only accepts 4-dimension tensors, with dimensions (batch_size,height,width,num_channels) '''
        with tf.Session(''):
            with self.assertRaises(ValueError):
               cl.corr([1, 2], [[1, 2], [3, 4]]).eval()
            with self.assertRaises(ValueError):
                self.assertRaises(cl.corr([1, 2], [1, 2, 3, 4]).eval(), ValueError)
            with self.assertRaises(ValueError):
                self.assertRaises(cl.corr([1, 2, 3], [[1, 2], [3, 4]]).eval(), ValueError)
    def test_raisesExceptionWithTooManyOffsets(self):
        ''' correlation only accepts up to 2601==51x51  offsets '''
        with tf.Session(''):
            with self.assertRaises(ValueError):
                # Current max_displacement/stride is 25 
                my_shape = (1,21,21,1)
                a = np.ones((my_shape))
                b = np.ones((my_shape))
                self.assertRaises(cl.corr(a,b,stride=1,max_displacement=50),ValueError);
            
    def test_correlationHardCoded(self):
        with tf.Session(''):
            batch_size = 1;
            width = 21;
            depth = 1;
            stride = 2;
            max_displacement = 20;
            expected_depth = (2*int(max_displacement/stride)+1)**2;
            my_shape = (batch_size,width,width,depth)
            result = cl.corr(np.ones(my_shape), np.ones(my_shape)).eval()
            self.assertEqual(result.shape[0], my_shape[0])
            self.assertEqual(result.shape[1], my_shape[1])
            self.assertEqual(result.shape[2], my_shape[2])
            self.assertEqual(result.shape[3], expected_depth)
            self.assertEqual(result[0,0,0,220], 1)
            self.assertEqual(result[0,0,0,0], 0)
            np.testing.assert_array_equal(result[0,:,:,220], np.ones((width,width)))
    
    def test_correlationGradientAHardCoded(self):
        with tf.Session('') as sess:
            batch_size = 1;
            height = 21;
            width = 21;
            depth = 1;
            stride = 2;
            max_displacement = 20;
            expected_depth = (2*int(max_displacement/stride)+1)**2;
            offsets = []
            for row_offset in np.arange(-max_displacement,max_displacement+1,stride):
                for col_offset in np.arange(-max_displacement,max_displacement+1,stride):
                     offsets.append((row_offset,col_offset)); 

            my_shape = (batch_size,height,width,depth)
            a = tf.placeholder(tf.float32, shape = my_shape)
            feed_a = np.ones(my_shape,dtype=np.float32)
            b = 2 * np.ones(my_shape,dtype=np.float32)
            result = cl.corr(a, b)
            self.assertEqual(int(result.shape[3]), expected_depth)

            # Check if it's aligned at all offsets
            for offset_index,offset in enumerate(offsets):
                result_slices  = result[:,:,:,offset_index]
                grad_a = tf.gradients(result_slices,a);            
                gradient_a = sess.run(grad_a,feed_dict={a : feed_a});
                row_offset = offset[0]
                col_offset = offset[1]
                a_row_begin = 0
                a_row_end = height-row_offset
                b_row_begin = row_offset
                b_row_end = height
                
                a_col_begin = 0
                a_col_end   = width-col_offset
                b_col_begin = col_offset
                b_col_end   = width
                
                if(row_offset < 0):
                   a_row_begin = -row_offset
                   a_row_end  = height
                   b_row_begin = 0
                   b_row_end = height+row_offset
                if(col_offset < 0):
                   a_col_begin = -col_offset
                   a_col_end  = width
                   b_col_begin = 0
                   b_col_end = width+col_offset
                final_height = a_row_end-a_row_begin
                final_width = a_col_end-a_col_begin
                np.testing.assert_array_equal(gradient_a[0][0,a_row_begin:a_row_end,a_col_begin:a_col_end,0], 2*np.ones((final_height,final_width)))

    def test_correlationGradientAHardCodedLarge(self):
        with tf.Session('') as sess:
            batch_size = 1;
            height = 41;
            width = 41;
            depth = 1;
            stride = 4;
            max_displacement = 24;
            expected_depth = (2*int(max_displacement/stride)+1)**2;
            offsets = []
            for row_offset in np.arange(-max_displacement,max_displacement+1,stride):
                for col_offset in np.arange(-max_displacement,max_displacement+1,stride):
                     offsets.append((row_offset,col_offset)); 

            my_shape = (batch_size,height,width,depth)
            a = tf.placeholder(tf.float32, shape = my_shape)
            feed_a = np.ones(my_shape,dtype=np.float32)
            b = 2 * np.ones(my_shape,dtype=np.float32)
            result = cl.corr(a, b,stride=stride,max_displacement=max_displacement)
            self.assertEqual(int(result.shape[3]), expected_depth)

            # Check if it's aligned at all offsets
            for offset_index,offset in enumerate(offsets):
                result_slices  = result[:,:,:,offset_index]
                grad_a = tf.gradients(result_slices,a);            
                gradient_a = sess.run(grad_a,feed_dict={a : feed_a});
                row_offset = offset[0]
                col_offset = offset[1]
                a_row_begin = 0
                a_row_end = height-row_offset
                b_row_begin = row_offset
                b_row_end = height
                
                a_col_begin = 0
                a_col_end   = width-col_offset
                b_col_begin = col_offset
                b_col_end   = width
                
                if(row_offset < 0):
                   a_row_begin = -row_offset
                   a_row_end  = height
                   b_row_begin = 0
                   b_row_end = height+row_offset
                if(col_offset < 0):
                   a_col_begin = -col_offset
                   a_col_end  = width
                   b_col_begin = 0
                   b_col_end = width+col_offset
                final_height = a_row_end-a_row_begin
                final_width = a_col_end-a_col_begin
                np.testing.assert_array_equal(gradient_a[0][0,a_row_begin:a_row_end,a_col_begin:a_col_end,0], 2*np.ones((final_height,final_width)))
    
    def test_correlationGradientBHardCoded(self):
        with tf.Session('') as sess:
            batch_size = 1;
            height = 21;
            width = 21;
            depth = 1;
            stride = 2;
            max_displacement = 20;
            expected_depth = (2*int(max_displacement/stride)+1)**2;
            stride = 2;
            max_displacement = 20;
            expected_depth = (2*int(max_displacement/stride)+1)**2;
            offsets = []
            for row_offset in np.arange(-max_displacement,max_displacement+1,stride):
                for col_offset in np.arange(-max_displacement,max_displacement+1,stride):
                     offsets.append((row_offset,col_offset)); 

            my_shape = (batch_size,height,width,depth)
            a = np.ones(my_shape,dtype=np.float32)
            feed_b = 2*np.ones(my_shape,dtype=np.float32)
            b = tf.placeholder(tf.float32, shape = my_shape)
            result = cl.corr(a, b)

            # Check if it's aligned at all offsets
            for offset_index,offset in enumerate(offsets):
                result_slices  = result[:,:,:,offset_index]
                grad_b = tf.gradients(result_slices,b);            
                gradient_b = sess.run(grad_b,feed_dict={b : feed_b});
                row_offset = offset[0]
                col_offset = offset[1]
                a_row_begin = 0
                a_row_end = height-row_offset
                b_row_begin = row_offset
                b_row_end = height
                
                a_col_begin = 0
                a_col_end   = width-col_offset
                b_col_begin = col_offset
                b_col_end   = width
                
                if(row_offset < 0):
                   a_row_begin = -row_offset
                   a_row_end  = height
                   b_row_begin = 0
                   b_row_end = height+row_offset
                if(col_offset < 0):
                   a_col_begin = -col_offset
                   a_col_end  = width
                   b_col_begin = 0
                   b_col_end = width+col_offset
                final_height = b_row_end-b_row_begin
                final_width = b_col_end-b_col_begin
                np.testing.assert_array_equal(gradient_b[0][0,b_row_begin:b_row_end,b_col_begin:b_col_end,0], np.ones((final_height,final_width)))
    
    def test_correlationRandom(self):
        with tf.Session(''):
            batch_size = 1;
            height = 21;
            width = 21;
            depth = 1;
            my_shape = (batch_size,height,width,depth)
            stride = 2;
            max_displacement = 20;
            expected_depth = (2*int(max_displacement/stride)+1)**2;
            offsets = []
            for row_offset in np.arange(-max_displacement,max_displacement+1,stride):
                for col_offset in np.arange(-max_displacement,max_displacement+1,stride):
                     offsets.append((row_offset,col_offset)); 

            for i in range(10):
                a_rand = np.random.randint(10, size = my_shape)
                b_rand = np.random.randint(10, size = my_shape)
                
                result = cl.corr(a_rand, b_rand,stride=stride,max_displacement=max_displacement).eval()
                self.assertEqual(result.shape[3], expected_depth)
                for offset_index, offset in enumerate(offsets):
                    row_offset = offset[0]
                    col_offset = offset[1]
                    a_row_begin = 0
                    a_row_end = height-row_offset
                    b_row_begin = row_offset
                    b_row_end = height
                    
                    a_col_begin = 0
                    a_col_end   = width-col_offset
                    b_col_begin = col_offset
                    b_col_end   = width
                    
                    if(row_offset < 0):
                       a_row_begin = -row_offset
                       a_row_end  = height
                       b_row_begin = 0
                       b_row_end = height+row_offset
                    if(col_offset < 0):
                       a_col_begin = -col_offset
                       a_col_end  = width
                       b_col_begin = 0
                       b_col_end = width+col_offset
                    a_slice = a_rand[:,a_row_begin:a_row_end,a_col_begin:a_col_end,:]
                    b_slice = b_rand[:,b_row_begin:b_row_end,b_col_begin:b_col_end,:];
                    result_rand_full = a_slice*b_slice 
                    result_rand = np.sum(result_rand_full,axis=-1)/depth
                    np.testing.assert_array_equal(result[0,a_row_begin:a_row_end,a_col_begin:a_col_end,offset_index], result_rand[0,:,:])

    def test_correlationGradientARandom(self):
        with tf.Session('') as sess:
            batch_size = 1;
            height = 21;
            width = 21;
            depth = 1;
            stride = 2;
            max_displacement = 20;
            expected_depth = (2*int(max_displacement/stride)+1)**2;
            stride = 2;
            max_displacement = 20;
            expected_depth = (2*int(max_displacement/stride)+1)**2;
            offsets = []
            for row_offset in np.arange(-max_displacement,max_displacement+1,stride):
                for col_offset in np.arange(-max_displacement,max_displacement+1,stride):
                     offsets.append((row_offset,col_offset)); 

            my_shape = (batch_size,height,width,depth)
            a = tf.placeholder(tf.float32, shape = my_shape)
            feed_a = np.random.randint(10,size=my_shape).astype(np.float32)
            b = 2 * np.random.randint(10,size=my_shape).astype(np.float32)
            result = cl.corr(a, b)

            # Check if it's aligned at all offsets
            for offset_index,offset in enumerate(offsets):
                row_offset = offset[0]
                col_offset = offset[1]
                a_row_begin = 0
                a_row_end = height-row_offset
                b_row_begin = row_offset
                b_row_end = height
                
                a_col_begin = 0
                a_col_end   = width-col_offset
                b_col_begin = col_offset
                b_col_end   = width
                
                if(row_offset < 0):
                   a_row_begin = -row_offset
                   a_row_end  = height
                   b_row_begin = 0
                   b_row_end = height+row_offset
                if(col_offset < 0):
                   a_col_begin = -col_offset
                   a_col_end  = width
                   b_col_begin = 0
                   b_col_end = width+col_offset
                result_slice = result[:,:,:,offset_index]
                grad_a = tf.gradients(result_slice,a);
                gradient_a = sess.run(grad_a,feed_dict={a : feed_a});
                np.testing.assert_array_equal(gradient_a[0][0,a_row_begin:a_row_end,a_col_begin:a_col_end,0], b[0,b_row_begin:b_row_end,b_col_begin:b_col_end,0])

                
    def test_correlationGradientBRandom(self):
        with tf.Session('') as sess:
            batch_size = 1;
            height = 21;
            width = 21;
            depth = 1;
            stride = 2;
            max_displacement = 20;
            expected_depth = (2*int(max_displacement/stride)+1)**2;
            stride = 2;
            max_displacement = 20;
            expected_depth = (2*int(max_displacement/stride)+1)**2;
            offsets = []
            for row_offset in np.arange(-max_displacement,max_displacement+1,stride):
                for col_offset in np.arange(-max_displacement,max_displacement+1,stride):
                     offsets.append((row_offset,col_offset)); 

            my_shape = (batch_size,height,width,depth)
            a = np.random.randint(10,size=my_shape).astype(np.float32)
            feed_b = np.random.randint(10,size=my_shape).astype(np.float32)
            b = tf.placeholder(tf.float32, shape = my_shape)
            result = cl.corr(a, b)

            # Check if it's aligned at all offsets
            for offset_index,offset in enumerate(offsets):
                row_offset = offset[0]
                col_offset = offset[1]
                a_row_begin = 0
                a_row_end = height-row_offset
                b_row_begin = row_offset
                b_row_end = height
                
                a_col_begin = 0
                a_col_end   = width-col_offset
                b_col_begin = col_offset
                b_col_end   = width
                
                if(row_offset < 0):
                   a_row_begin = -row_offset
                   a_row_end  = height
                   b_row_begin = 0
                   b_row_end = height+row_offset
                if(col_offset < 0):
                   a_col_begin = -col_offset
                   a_col_end  = width
                   b_col_begin = 0
                   b_col_end = width+col_offset
                result_slice = result[:,:,:,offset_index]
                grad_b = tf.gradients(result_slice,b);
                gradient_b = sess.run(grad_b,feed_dict={b : feed_b});
                np.testing.assert_array_equal(a[0,a_row_begin:a_row_end,a_col_begin:a_col_end,0], gradient_b[0][0,b_row_begin:b_row_end,b_col_begin:b_col_end,0])
                  
                
if __name__ == '__main__':
    suite=unittest.TestLoader().loadTestsFromTestCase(CorrelationOpTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
