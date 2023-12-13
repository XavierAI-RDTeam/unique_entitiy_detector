from DuplicateDetectorTrainer import DuplicateDetectorTrainer

if __name__ == '__main__':
    trainer = DuplicateDetectorTrainer()

    trainer.trainBaseModel()
    trainer.trainExtendedModel()