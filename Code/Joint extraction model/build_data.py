import os
import utils
import parsers
from sklearn.externals import joblib
import os.path

class build_data():
    def __init__(self,fname):


        config_file=parsers.read_properties(fname)
        self.config_fname=fname

        self.filename_embeddings = config_file.getProperty("filename_embeddings")
        self.filename_train=config_file.getProperty("filename_train")
        self.filename_test=config_file.getProperty("filename_test")
        self.filename_dev=config_file.getProperty("filename_dev")
        
        self.train_id_docs = parsers.readHeadFile(self.filename_train)
        self.dev_id_docs = parsers.readHeadFile( self.filename_dev)
        self.test_id_docs = parsers.readHeadFile(self.filename_test)
        dataset_documents = []
        dataset_documents.extend(self.train_id_docs)
        dataset_documents.extend(self.dev_id_docs)
        dataset_documents.extend(self.test_id_docs)
        self.dataset_set_characters = utils.getCharsFromDocuments(dataset_documents)
        self.dataset_set_bio_tags, self.dataset_set_ec_tags = utils.getEntitiesFromDocuments(dataset_documents)
        self.dataset_set_relations = utils.getRelationsFromDocuments(dataset_documents)



        if os.path.isfile(self.filename_embeddings+".pkl")==False:
            self.wordvectors, self.representationsize, self.words = utils.readWordvectorsNumpy(self.filename_embeddings, isBinary=True if self.filename_embeddings.endswith(".bin") else False)
            self.wordindices = utils.readIndices(self.filename_embeddings,
                                                 isBinary=True if self.filename_embeddings.endswith(".bin") else False)
            joblib.dump((self.wordvectors, self.representationsize, self.words,self.wordindices), self.filename_embeddings+".pkl")

        else:
            self.wordvectors, self.representationsize, self.words,self.wordindices = joblib.load(self.filename_embeddings + ".pkl")  # loading is faster



        parsers.preprocess(self.train_id_docs, self.wordindices, self.dataset_set_characters,
                                                self.dataset_set_bio_tags, self.dataset_set_ec_tags, self.dataset_set_relations)
        parsers.preprocess(self.dev_id_docs, self.wordindices, self.dataset_set_characters,
                           self.dataset_set_bio_tags, self.dataset_set_ec_tags, self.dataset_set_relations)
        parsers.preprocess(self.test_id_docs, self.wordindices, self.dataset_set_characters,
                           self.dataset_set_bio_tags, self.dataset_set_ec_tags, self.dataset_set_relations)



        self.nepochs = int(config_file.getProperty("nepochs"))
        self.optimizer = config_file.getProperty("optimizer")
        self.activation =config_file.getProperty("activation")
        self.learning_rate =float(config_file.getProperty("learning_rate"))
        self.gradientClipping = utils.strToBool(config_file.getProperty("gradientClipping"))
        self.nepoch_no_imprv = int(config_file.getProperty("nepoch_no_imprv"))
        self.use_dropout = utils.strToBool(config_file.getProperty("use_dropout"))
        self.ner_loss = config_file.getProperty("ner_loss")
        self.ner_classes = config_file.getProperty("ner_classes")
        self.use_chars = utils.strToBool(config_file.getProperty("use_chars"))
        self.use_adversarial = utils.strToBool(config_file.getProperty("use_adversarial"))
        self.use_att = utils.strToBool(config_file.getProperty("use_att"))



        self.dropout_embedding = float(config_file.getProperty("dropout_embedding"))
        self.dropout_lstm = float(config_file.getProperty("dropout_lstm"))
        self.dropout_lstm_output = float(config_file.getProperty("dropout_lstm_output"))
        self.dropout_att = float(config_file.getProperty("dropout_att"))
        self.dropout_fcl_ner = float(config_file.getProperty("dropout_fcl_ner"))
        self.dropout_fcl_rel = float(config_file.getProperty("dropout_fcl_rel"))
        self.hidden_size_lstm =int(config_file.getProperty("hidden_size_lstm"))
        self.hidden_size_n1 = int(config_file.getProperty("hidden_size_n1"))
        self.num_lstm_layers = int(config_file.getProperty("num_lstm_layers"))
        self.char_embeddings_size = int(config_file.getProperty("char_embeddings_size"))
        self.hidden_size_char = int(config_file.getProperty("hidden_size_char"))
        self.label_embeddings_size = int(config_file.getProperty("label_embeddings_size"))
        self.alpha = float(config_file.getProperty("alpha"))

        self.evaluation_method =config_file.getProperty("evaluation_method")
        self.root_node=utils.strToBool(config_file.getProperty("root_node"))

        self.shuffle=False
        self.batchsize=1







