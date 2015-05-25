# Install various packages and python modules for data analysis
Exec {
  path => '/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin',
}

$deps = ['python3-pip', 'gfortran', 'liblapack-dev', 'libfreetype6',
  'libfreetype6-dev', 'ipython3-notebook', 'weka', 'build-essential']

package {$deps:
  ensure => latest,
}

exec {'module-deps':
  command => 'pip3 install pyparsing tornado pytz dateutils numpy scipy pandas',
  unless  => 'pip3 freeze | grep numpy',
  require => Package[$deps],
}

exec {'numpy':
  command => 'pip3 install numpy',
  unless  => 'pip3 freeze | grep numpy',
  require => Exec['module-deps'],
}

exec {'scipy':
  command => 'pip3 install scipy',
  unless  => 'pip3 freeze | grep scipy',
  timeout => 600,
  require => Exec['numpy'],
}

exec {'pandas':
  command => 'pip3 install pandas',
  unless  => 'pip3 freeze | grep pandas',
  require => Exec['scipy'],
}

file {'/usr/include/ft2build.h':
  ensure  => link,
  target  => '/usr/include/freetype2/ft2build.h',
  require => Exec['module-deps'],
}

exec {'matplotlib':
  command => 'pip3 install matplotlib',
  unless  => 'pip3 freeze | grep matplotlib',
  require => File['/usr/include/ft2build.h'],
}

exec {'sklearn':
  command => 'pip3 install scikit-learn',
  unless  => 'pip3 freeze | grep scikit-learn',
  require => Exec['matplotlib'],
}
