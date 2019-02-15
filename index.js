import './style'
import { Component } from 'preact'
import * as tf from '@tensorflow/tfjs'

import {/*IMAGE_H, IMAGE_W,*/ MnistData} from './mnist'
import {drawCanvas, getImages, showTestResults} from './ui.js'
import mnistDCGAN from './mnistgan'

import Samples from './components/samples'
import Canvas from './components/canvas'


export default class App extends Component {

	state = {
		data: null,
		dcgan: null,
		toggle: true,
		samples: null,
		epoch: 0
	}

	train = async (epochs = 2000) => {
		const g = this.state.dcgan
		for (let i = 0; i < epochs; i++) {
			await tf.nextFrame()
			if(!this.state.training ) {break}
			
			let noise =	g.gan.noise(32)
			let fakes = g.gen.predict(noise)
			let real = this.state.data.nextTrainBatch(32).xs.reshapeAs(fakes)
			let x = tf.concat([real, fakes])
			let y = tf.concat([g.ONES_CAP, g.ZEROS])
			
			let dLoss = await g.discriminator.trainOnBatch(x, y)			
			console.log('discriminator done')
			await tf.nextFrame()

			y = g.ONES
			noise = g.gan.noise(32)

			let aLoss = await g.adversarial.trainOnBatch(noise, y)
			console.log('adversarial done')
						
			if(i % 5 === 0 && i != 0){
				this.setState((state, props) => {
					const p = g.gen.predict(this.state.noise)
					console.log('samples update')
					return {samples: p}
				})				
			}
			this.setState((state, props) => ({epoch: this.state.epoch + 1}))
			await tf.nextFrame()
			console.log(dLoss, aLoss)
			
		}
		console.log('End!')
	}

	startTrain = () => {
		console.log('Start...')
		this.setState((state, props) => {
			return {training: true}
		})
		this.train()		
	}

	stopTrain = () => {
		console.log('Stopping...')		
		this.setState({
			training: false
		})
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
					batchSize: 32,
					data
				})
			})
		}).then(() => {
			const noise = this.state.dcgan.gan.noise(9)
			this.setState({
				noise
			})
		}).then(() => {
			this.setState({
				samples: this.state.dcgan.gen.predict(this.state.noise)
			})
		})
	}

	toggle = () => {
		this.setState({toggle: !this.state.toggle})
	}

	renewSamples = () => {
		this.setState((state, props) => {
			const noise = this.state.dcgan.gan.noise(9)
			const p = this.state.dcgan.gen.predict(noise)
			return {samples: p}
		})
	}

	render({}, {epoch, samples}) {
		return (
			<div>
				<h1>Hello, World!</h1>
				<button onClick={this.startTrain}>Train</button>
				<button onClick={this.stopTrain}>Stop</button>
				<button onClick={this.renewSamples}>Samples</button>
				<button onClick={this.toggle}>toggle</button>
				{/* {samples && <Samples examples={samples} />} */}
				<div>
					<h3>Samples</h3>
					<h4>{epoch}</h4>
					<div id="testSamples">
						{samples && getImages(samples).map(img => <Canvas data={img}/>)}            
						{/* {showTestResults(examples)} */}
					</div>
				</div>
			</div>
		)
	}
}
