from logging import DEBUG, Formatter, getLogger, StreamHandler
import config
import spectral_network_generator

logger = getLogger(__name__)
logger.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter('\t'.join(['%(asctime)s', '[%(levelname)s]', '%(name)s(Line:%(lineno)d)', '%(message)s']))
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


if __name__ == '__main__':
    config_list = config.read_config_file(path='./config.ini')

    list_conf_id_vs_spec_cmpd_sim_correlation = []

    for conf in config_list:
        logger.info(conf.id)
        conf_id_vs_correlation = []

        logger.debug('######################')
        logger.debug(conf.__dict__)
        spectral_network_generator.generate_spectral_network(conf, _logger=logger)

