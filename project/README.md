# Bringing up development environment #
* Ensure Vagrant/VirtualBox are installed
* Customize [config.rb](config.rb) with the appropriate values
  for allocating RAM and CPUs to the virtual machine. At least
  1024 MB RAM is recommended for installing pandas.
* Execute `vagrant up` from the project root
* `vagrant ssh` into the box, then execute `start`, an alias
  to start the iPython Notebook.
* Browse to [http://localhost:8888](http://localhost:8888) in your browser

# Documentation #
* [Scikit Learn Intro Video Series](http://blog.kaggle.com/2015/04/08/new-video-series-introduction-to-machine-learning-with-scikit-learn/)
* [Numpy, Pandas, Matplotlib Cheat Sheet](docs/Quandl+-+Pandas,+SciPy,+NumPy+Cheat+Sheet.pdf)

