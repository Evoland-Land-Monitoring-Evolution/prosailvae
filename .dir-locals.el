((python-mode
  . ((eglot-workspace-configuration
      . ((pylsp
	  . ((configurationSources . ["flake8"])
	     (plugins
              (autopep8 (enabled . :json-false))
              (pylsp_black (enabled . t))
	      (pyls_isort (enabled . t))))))))))
