
# ----------------------------

def read_json_entry(config, key, default = None):
    
    if default == None:
        try:
            value = config[key]
        except KeyError:
            raise RuntimeError("{} is a required entry in the JSON configuration file.".format(key))
    else:
        value = config.get(key, default)

    return value

# ----------------------------

