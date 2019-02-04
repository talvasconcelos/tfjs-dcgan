import * as tf from '@tensorflow/tfjs'
import DCGAN from './dcgan'

class mnistDCGAN {
    constructor(opts){
        opts = opts || {}
        this.imgSize = opts.imgSize || 64
        this.imgC = opts.imgC || 3
        this.batchSize = opts.batchSize || 64 //batch size
        this.epochs = opts.epochs || 50
        this.save = opts.save || 10

        this.real = opts.real

        this.gan = new DCGAN({
            imgSize: this.imgSize,
            imgC: this.imgC,
            batchSize: this.batchSize
        })
        this.discriminator = this.gan.dicriminatorModel()
        this.adversarial = this.gan.adversarialModel()
        this.gen = this.gan.generator()

        this.ONES = tf.ones([this.batchSize, 1])
        this.ONES_CAP = tf.ones([this.batchSize, 1]).mul(tf.scalar(0.9))
        this.ZEROS = tf.randomUniform([this.batchSize, 1], 0, 0.1)
    }

}

export default mnistDCGAN