import './style'
import { Component } from 'preact'
import * as tf from '@tensorflow/tfjs'

import {/*IMAGE_H, IMAGE_W,*/ MnistData} from './mnist'
import mnistDCGAN from './mnistgan'

import Samples from './components/samples'

export default class App extends Component {

	state = {
		data: null,
		dcgan: null
	}

	train = async (epochs = 5) => {
		const g = this.state.dcgan
		await this.setState({
			samples: g.gen.predict(this.state.noise)
		})
		await tf.nextFrame()
		for (let i = 0; i < epochs; i++) {
			let noise =	g.gan.noise(64)
			let fakes = g.gen.predict(noise)
			let real = this.state.data.nextTrainBatch(64).xs.reshapeAs(fakes)
			let x = tf.concat([real, fakes])
			let y = tf.concat([g.ONES_CAP, g.ZEROS])
			
			let dLoss = await g.discriminator.trainOnBatch(x, y)
			await tf.nextFrame()

			y = g.ONES
			noise = g.gan.noise(64)

			// console.log(y.shape)
			// console.log(noise.shape)

			let aLoss = await g.adversarial.trainOnBatch(noise, y)
			// console.log(aLoss)
			
			await this.setState({
				samples: g.gen.predict(this.state.noise)
			})
			await tf.nextFrame()
			console.log(dLoss, aLoss)
		}
	}

	startTrain = () => {
		console.log('Start...')
		
		this.train()
	}

	componentDidMount = () => {
		const data = new MnistData()
		data.load()
		.then(() => {
			this.setState({
				data,
				dcgan: new mnistDCGAN({
					imgSize: 28,
					imgC: 1,
					batchSize: 64,
					data
				})
			})
			}).then(() => {
				const noise = this.state.dcgan.gan.noise(9)
				this.setState({
					noise
				})
			})
	}

	render({}, {samples}) {
		return (
			<div>
				<h1>Hello, World!</h1>
				<button onClick={this.startTrain}>Train</button>
				{samples && <Samples examples={samples} />}
			</div>
		)
	}
}
