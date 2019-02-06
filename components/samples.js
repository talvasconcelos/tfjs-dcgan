import {drawCanvas, getImages, showTestResults} from '../ui.js'
import Canvas from './canvas'

const Samples = ({examples}) => {
    
    return (
        <div>
            <h3>Samples</h3>
            <div id="testSamples">
                {getImages(examples).map(img => <Canvas data={img}/>)}            
                {/* {showTestResults(examples)} */}
            </div>
        </div>
    )
}

export default Samples