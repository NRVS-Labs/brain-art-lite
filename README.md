# brain-art-lite variant/streamlit 

An exploration into pairing Creative Neural Networks with actual neural networks in the human brain using interpreted brain activity

## Roadmap

We are actively working on improving the intentionality of Brain Generated Artwork! We are working towards in the future adding:
* Real time artwork representative of real time biosignals
* Emotion recognition and interpretation
* Experimenting with alternative Art Generation Models
* Implementation in other frameworks
* Custom art generation programs designed / written by the user

## Acknowledgements

This project is a update to the BrainArt project built at Union Neurotech (`https://github.com/Union-Neurotech/BrainArt`) and the Telepathic Polluck project created for the 2022 Brain.io hackathon. The original project is accessible here: `https://github.com/LeonardoFerrisi/telepathic-polluck`

The CPPN (Compositional Pattern Producing Networks) art generation model was based off of and inspired by work from a variety of excellent sources on the topic listed below:
- https://janhuenermann.com/blog/abstract-art-with-ml/
- https://kwj2104.github.io/2018/cppngan/
- https://towardsdatascience.com/understanding-compositional-pattern-producing-networks-810f6bef1b88
- https://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/
- https://github.com/ex-punctis/abstract-art-cppn
- https://www.expunctis.com/2020/01/19/Abstract-art.html
- https://medium.com/@tuanle618/generate-abstract-random-art-with-a-neural-network-ecef26f3dd5f
- https://github.com/tuanle618/neural-net-random-art	
- https://github.com/tuanle618/neural-net-random-art/blob/master/nb_random_art.ipynb
- https://github.com/hardmaru/cppn-tensorflow/tree/master

# Notes
* It seems that networks of smaller layers consistently produce smoother, more appealing images.
* The input scalar values are placeholders for when we eventually begin baking in additional quantified states such as emotion. These scalars act as weight and bias modifiers.
    * An additional way scalars can be modified (to be implemented) is via post-image generation modification via application of a variety of 'filters' which would be determined by additional inputs. Would certainly add more control / intentionality to what values result in visually.