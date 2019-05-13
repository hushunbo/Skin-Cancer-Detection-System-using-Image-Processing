function [Cmat,Accuracy]= confusion_matrix(predicted,labels,classes_names)

C=confusionmat(labels,predicted);
L=length(unique(labels));
for i=1:L
    Cmat(i,:)=C(i,:)./sum(C(i,:));
end
figure('visible','on');
imagesc(Cmat);colormap(flipud(summer));caxis([0,20])
textstr=num2str(Cmat(:),'%0.2f');
textstr=strtrim(cellstr(textstr));
[x,y]=meshgrid(1:L);
hstrg=text(x(:),y(:),textstr(:),'HorizontalAlignment','center','FontSize',16,'FontName','Times New Roman');
midvalue=mean(get(gca,'Clim'));
textColors=repmat(Cmat(:)>midvalue,1,3);
set(hstrg,{'color'},num2cell(textColors,2));
set(gca,'XTick',1:L,'XTickLabel',classes_names,'YTick',1:L,'YTickLabel',classes_names,'TickLength',[0,0],'FontSize',13,'FontName','Times New Roman');
colorbar;
%Accuracy=mean(diag(Cmat))*100
end